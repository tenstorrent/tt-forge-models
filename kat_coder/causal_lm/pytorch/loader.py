# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KAT-Coder model loader implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available KAT-Coder model variants for causal language modeling."""

    KAT_CODER_V2_5_DEV = "KAT-Coder-V2.5-Dev"


class ModelLoader(ForgeModel):
    """KAT-Coder model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.KAT_CODER_V2_5_DEV: LLMModelConfig(
            pretrained_model_name="Kwaipilot/KAT-Coder-V2.5-Dev",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KAT_CODER_V2_5_DEV

    sample_text = "Who are you?"

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
            model="KAT-Coder",
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

        # KAT-Coder's tokenizer ships without a pad token; reuse EOS so padding works.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the KAT-Coder model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The KAT-Coder model for causal language modeling.
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
        # Force use_cache=False so the forward output does not include a hybrid
        # DynamicCache with LinearAttentionLayer entries (no .keys/.values). The
        # runner's pytree comparator (_cache_to_legacy) only handles standard
        # KV layers. Same pattern as qwen_3_5/causal_lm.
        model.config.use_cache = False
        self.config = model.config
        self.model = model
        print("self, model:", self.model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the KAT-Coder model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        if self.tokenizer.chat_template is not None:
            conversation = [{"role": "user", "content": self.sample_text}]
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self.sample_text
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
        # KAT-Coder is a Qwen3_5Moe hybrid, so this mirrors the
        # qwen_3_5/causal_lm loader. Most decoder layers use GatedDeltaNet
        # linear attention (``linear_attn``); every 4th layer uses full
        # attention (``self_attn``). Every layer's MLP is a
        # Qwen3_5MoeSparseMoeBlock (fused routed experts + a dense shared
        # expert).
        #
        # NOTE: the fused routed experts (mlp.experts.gate_up_proj /
        # down_proj) are sharded on the expert dimension by the custom MoE
        # injection (get_tt_moe_shard_specs / inject_custom_moe), which wraps
        # them in an ``sdy.manual_computation``. Sharding them again here binds
        # the "model" axis twice and fails the StableHLO pipeline with
        # "sdy.all_slice ... already bound by a parent sdy.manual_computation",
        # so we deliberately do NOT set specs for them. The router (gate) and
        # shared_expert_gate also stay replicated.
        shard_specs = {}
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                # Shared expert is a dense MLP: column-parallel gate/up,
                # row-parallel down.
                shared = mlp.shared_expert
                shard_specs[shared.gate_proj.weight] = ("model", "batch")
                shard_specs[shared.up_proj.weight] = ("model", "batch")
                shard_specs[shared.down_proj.weight] = ("batch", "model")
            else:
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
                # KAT-Coder V2.5 has only 2 KV heads (k/v_proj out=512,
                # head_dim 256), which cannot be split across the "model" axis.
                # Shard the contracted (input) dim instead: q/k/v on
                # ("batch", "model"), o_proj on ("model", "batch") — the same
                # scheme qwen_3_5 uses for its low-KV-head variant.
                sa = layer.self_attn
                shard_specs[sa.q_proj.weight] = ("batch", "model")
                shard_specs[sa.k_proj.weight] = ("batch", "model")
                shard_specs[sa.v_proj.weight] = ("batch", "model")
                shard_specs[sa.o_proj.weight] = ("model", "batch")
            elif hasattr(layer, "linear_attn"):
                # GatedDeltaNet: shard the input projections' head dim on
                # "model" and contract it back on out_proj. conv1d weight stays
                # replicated (channel split does not align with the fused-qkv
                # boundaries under Shardy); dt_bias / A_log follow the heads.
                la = layer.linear_attn
                shard_specs[la.in_proj_qkv.weight] = ("model", "batch")
                if hasattr(la, "conv1d"):
                    shard_specs[la.conv1d.weight] = (None, None, None)
                shard_specs[la.in_proj_z.weight] = ("model", "batch")
                shard_specs[la.in_proj_a.weight] = ("model", "batch")
                shard_specs[la.in_proj_b.weight] = ("model", "batch")
                shard_specs[la.out_proj.weight] = ("batch", "model")
                if hasattr(la, "dt_bias"):
                    shard_specs[la.dt_bias] = ("model",)
                if hasattr(la, "A_log"):
                    shard_specs[la.A_log] = ("model",)

        shard_specs[model.model.embed_tokens.weight] = ("model", "batch")
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_activation_shard_spec(self, model):
        """Sharding constraints for intermediate ACTIVATIONS (not weights).

        The gated-delta block's fused ``in_proj_qkv`` is sharded contiguously on
        the "model" axis; the subsequent ``torch.split`` into [Q, K, V] cuts that
        sharded axis at points that don't align with the per-device boundaries,
        which miscompiles under Shardy and scrambles q/k/v before the recurrence
        (full-model PCC collapses). Replicating the conv output before the split
        makes the split run on correct data.
        """
        constraints = {}
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                constraints[layer.linear_attn.conv1d] = None
        return constraints

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config