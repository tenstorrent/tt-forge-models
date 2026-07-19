# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek MoE model loader implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    """Available DeepSeek Chat model variants for causal language modeling."""

    DEEPSEEK_LITE_CHAT = "DEEPSEEK_V2_LITE_CHAT"
    DEEPSEEK_CHAT = "DEEPSEEK_CODER_V2_LITE_INSTRUCT"
    DEEPSEEK_V2_CHAT = "DEEPSEEK_V2_CHAT"


class ModelLoader(ForgeModel):
    """DeepSeek Chat model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_LITE_CHAT: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
            max_length=256,
        ),
        ModelVariant.DEEPSEEK_CHAT: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            max_length=256,
        ),
        ModelVariant.DEEPSEEK_V2_CHAT: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-V2-Chat",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_LITE_CHAT

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
            model="DeepSeek-V2-Lite",
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

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepSeek Chat model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DeepSeek Chat model for causal language modeling.
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
        # transformers bug (e.g. 5.5.x): DeepseekV2Moe.route_tokens_to_experts
        # uses self.num_experts for group_limited_greedy routing (DeepSeek-V2 /
        # DeepSeek-V2-Chat), but __init__ never sets it. Lite variants use
        # topk_method="greedy" and skip this path. Fixed upstream by moving
        # routing onto DeepseekV2TopkRouter.
        for module in model.modules():
            if (
                type(module).__name__ == "DeepseekV2Moe"
                and not hasattr(module, "num_experts")
            ):
                module.num_experts = module.config.n_routed_experts
        model.eval()
        self.config = model.config
        self.model = model
        print("model loaded", model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DeepSeek Chat model with this instance's variant settings.

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
            # DeepSeek-V2 uses Multi-head Latent Attention (MLA) rather than
            # separate q/k/v projections. q_proj / q_b_proj and kv_b_proj
            # (which recovers per-head K/V from the shared latent) are
            # column-sharded over heads; o_proj is row-sharded to match.
            # kv_a_proj_with_mqa (and q_a_proj when Q-LoRA is enabled) produce
            # a shared latent every device needs in full, so they stay
            # replicated.
            sa = layer.self_attn
            if getattr(sa, "q_lora_rank", None) is None:
                shard_specs[sa.q_proj.weight] = ("model", "batch")
            else:
                shard_specs[sa.q_b_proj.weight] = ("model", "batch")
            shard_specs[sa.kv_b_proj.weight] = ("model", "batch")
            shard_specs[sa.o_proj.weight] = ("batch", "model")

            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                # Sparse MoE layer: whole experts are distributed across
                # devices (expert-parallel). The router must see all experts
                # to make correct top-k choices, so it stays replicated.
                shard_specs[mlp.experts.gate_up_proj] = ("model", None, None)
                shard_specs[mlp.experts.down_proj] = ("model", None, None)

                shared = getattr(mlp, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.up_proj.weight] = ("model", "batch")
                    shard_specs[shared.gate_proj.weight] = ("model", "batch")
                    shard_specs[shared.down_proj.weight] = ("batch", "model")
            else:
                # Dense MLP (first layer only, before MoE kicks in).
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs
