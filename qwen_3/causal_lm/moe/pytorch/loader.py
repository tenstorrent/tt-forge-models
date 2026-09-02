# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Next MoE model loader implementation.

Qwen3-Next-80B-A3B uses a hybrid decoder: 3 Gated DeltaNet (linear attention)
layers followed by 1 gated full-attention layer, repeated, with a sparse MoE
MLP on every layer (512 experts, 10 active, 1 shared). Native in transformers
via ``Qwen3NextForCausalLM`` (repo default 5.5.1).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

from .....base import ForgeModel
from .....tools.conv_overrides import slice_depthwise_conv1d_channels
from .....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen3-Next MoE model variants for causal language modeling."""

    QWEN3_NEXT_80B_A3B_INSTRUCT = "Qwen3-Next-80B-A3B-Instruct"


class ModelLoader(ForgeModel):
    """Qwen3-Next MoE model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_NEXT_80B_A3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_NEXT_80B_A3B_INSTRUCT

    sample_text = "Give me a short introduction to large language model."

    # Gated DeltaNet conv is Conv1d(8192, 8192, k=4, groups=8192). ttnn can
    # only DRAM-slice height/width; a short prompt (output dim 21) has no
    # spatial split and overflows L1 (~1.4MB free). Chunk 1024 is the verified
    # fallback from Qwen 3.6 / KAT-Coder / Jamba on the same 4-chip mesh.
    CONV_CHANNEL_CHUNK = 1024

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen3-Next MoE",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen3-Next MoE model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen3-Next MoE model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model.config.use_cache = False
        slice_depthwise_conv1d_channels(model, self.CONV_CHANNEL_CHUNK)
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen3-Next MoE model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        conversation = [{"role": "user", "content": self.sample_text}]
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)

        assert (
            self.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                # Sparse MoE (Qwen3NextSparseMoeBlock / Qwen2MoeExperts): the
                # routed experts' fused 3D weights are sharded on the expert
                # dimension by get_tt_moe_shard_specs (inject_custom_moe). The
                # router (mlp.gate) and shared_expert_gate stay replicated so
                # every device can score all experts before dispatch. The
                # always-on shared expert is a dense MLP.
                shared = mlp.shared_expert
                shard_specs[shared.gate_proj.weight] = ("model", "batch")
                shard_specs[shared.up_proj.weight] = ("model", "batch")
                shard_specs[shared.down_proj.weight] = ("batch", "model")
            else:
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
                # Gated full attention (every 4th layer). Only 2 KV heads, so
                # shard the hidden dim rather than heads — same layout as
                # Qwen 3.5 35B-A3B. q_proj is 2x wide (query + sigmoid gate).
                sa = layer.self_attn
                shard_specs[sa.q_proj.weight] = ("batch", "model")
                shard_specs[sa.k_proj.weight] = ("batch", "model")
                shard_specs[sa.v_proj.weight] = ("batch", "model")
                shard_specs[sa.o_proj.weight] = ("model", "batch")
            elif hasattr(layer, "linear_attn"):
                # Gated DeltaNet: fused in_proj_qkvz / in_proj_ba, depthwise
                # conv1d, row-parallel out_proj. conv1d stays replicated so the
                # later Q/K/V split is not cut across device boundaries.
                la = layer.linear_attn
                shard_specs[la.in_proj_qkvz.weight] = ("model", "batch")
                shard_specs[la.in_proj_ba.weight] = ("model", "batch")
                if hasattr(la, "conv1d"):
                    shard_specs[la.conv1d.weight] = (None, None, None)
                shard_specs[la.out_proj.weight] = ("batch", "model")
                if hasattr(la, "dt_bias"):
                    shard_specs[la.dt_bias] = ("model",)
                if hasattr(la, "A_log"):
                    shard_specs[la.A_log] = ("model",)

        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_activation_shard_spec(self, model):
        """Sharding constraints for intermediate activations (not weights).

        The gated-delta block concatenates Q/K/V, runs depthwise conv1d, then
        splits. Replicating the conv output keeps that split aligned, matching
        the Qwen 3.5 hybrid-attention loader.
        """
        constraints = {}
        for layer in model.model.layers:
            if getattr(layer, "layer_type", None) == "linear_attention":
                constraints[layer.linear_attn.conv1d] = None
        return constraints
