# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AI21 Jamba Mini 1.7 (ai21labs/AI21-Jamba-Mini-1.7) model loader.

Jamba is a hybrid decoder: attention layers, Mamba layers, and MoE FFNs
(``JambaAttentionDecoderLayer`` / ``JambaMambaDecoderLayer`` with either
``JambaMLP`` or ``JambaSparseMoeBlock``). CUDA mamba-ssm kernels are disabled
so the eager path compiles on TT.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

from ...base import ForgeModel
from ...tools.conv_overrides import slice_depthwise_conv1d_channels
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available AI21 Jamba model variants for causal language modeling."""

    AI21_JAMBA_MINI_1_7 = "AI21-Jamba-Mini-1.7"


class ModelLoader(ForgeModel):
    """AI21 Jamba Mini 1.7 loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.AI21_JAMBA_MINI_1_7: LLMModelConfig(
            pretrained_model_name="ai21labs/AI21-Jamba-Mini-1.7",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AI21_JAMBA_MINI_1_7

    sample_text = "Who are you?"

    # Mamba depthwise conv is Conv1d(8192, 8192, k=4, groups=8192). ttnn can
    # only DRAM-slice height/width; a short prompt (output dim 18) has no
    # spatial split and overflows L1 (~1.4MB free). Chunk 1024 is the verified
    # fallback from Qwen 3.6 / KAT-Coder on the same 4-chip mesh.
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
            model="AI21 Jamba",
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
        """Load and return the Jamba Mini 1.7 model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: JambaForCausalLM in eval mode.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {
            # Eager attention + slow Mamba path: mamba-ssm CUDA kernels are not
            # available on TT, and flash-attn is not the compile target.
            "attn_implementation": "eager",
            "use_mamba_kernels": False,
            # transformers defaults JambaExperts to the "grouped_mm" path, which
            # calls aten::_grouped_mm. That op only has a CompositeExplicitAutograd
            # kernel, so on an XLA tensor it runs the host implementation against a
            # device pointer and segfaults instead of falling back or lowering.
            "experts_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()
        slice_depthwise_conv1d_channels(model, self.CONV_CHANNEL_CHUNK)
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for Jamba Mini 1.7.

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

        attn_heads = self.config.num_attention_heads
        kv_heads = getattr(self.config, "num_key_value_heads", attn_heads)
        model_axis = mesh_shape[1]
        if attn_heads % model_axis != 0:
            raise ValueError(
                f"Cannot evenly distribute {attn_heads} attention heads "
                f"across model axis size {model_axis}"
            )
        if kv_heads % model_axis != 0:
            raise ValueError(
                f"Cannot evenly distribute {kv_heads} KV heads "
                f"across model axis size {model_axis}"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP for Jamba attention / MLP, expert-parallel MoE.

        Attention layers shard Q/K/V column-wise and O row-wise. Dense FFNs
        (``JambaMLP``) use the usual gate/up/down map. MoE FFNs
        (``JambaSparseMoeBlock``) store experts as fused 3D tensors; the expert
        axis is split on ``model`` and the router stays replicated. Mamba
        ``in_proj`` / ``out_proj`` follow column/row MLP layout; depthwise
        ``conv1d`` is replicated.
        """
        shard_specs = {}
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                sa = layer.self_attn
                shard_specs[sa.q_proj.weight] = ("model", "batch")
                shard_specs[sa.k_proj.weight] = ("model", "batch")
                shard_specs[sa.v_proj.weight] = ("model", "batch")
                shard_specs[sa.o_proj.weight] = ("batch", "model")
            elif hasattr(layer, "mamba"):
                mamba = layer.mamba
                shard_specs[mamba.in_proj.weight] = ("model", "batch")
                shard_specs[mamba.out_proj.weight] = ("batch", "model")
                shard_specs[mamba.dt_proj.weight] = ("model", "batch")
                shard_specs[mamba.conv1d.weight] = (None, None, None)
                shard_specs[mamba.A_log] = ("model", None)
                shard_specs[mamba.D] = ("model",)

            ff = layer.feed_forward
            if hasattr(ff, "experts"):
                # JambaExperts: (num_experts, 2*intermediate, hidden) /
                # (num_experts, hidden, intermediate). Router stays replicated.
                shard_specs[ff.experts.gate_up_proj] = ("model", None, None)
                shard_specs[ff.experts.down_proj] = ("model", None, None)
            else:
                shard_specs[ff.gate_proj.weight] = ("model", "batch")
                shard_specs[ff.up_proj.weight] = ("model", "batch")
                shard_specs[ff.down_proj.weight] = ("batch", "model")

        shard_specs[model.model.embed_tokens.weight] = ("model", "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
