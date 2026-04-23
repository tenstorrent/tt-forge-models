# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.2 Speciale model loader implementation for causal language modeling.

Uses reduced MoE configuration for testing since the full 671B parameter
model is too large to load directly.
"""

from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, DeepseekV3Config

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """DeepSeek V3.2 Speciale model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "deepseek-ai/DeepSeek-V3.2-Speciale"
        self.tokenizer = None
        self.text = "Please reason step by step. What is 25 multiplied by 16?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V3.2-Speciale",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # deepseek_v32 model type is not in released transformers; use DeepseekV3
        # as the base architecture with reduced dimensions for testing.
        config = DeepseekV3Config(
            num_hidden_layers=self.num_layers if self.num_layers is not None else 6,
            num_attention_heads=16,
            hidden_size=1024,
            num_key_value_heads=16,
            intermediate_size=1024 * 4,
            num_experts_per_tok=2,
            q_lora_rank=256,
            moe_intermediate_size=512,
            n_routed_experts=4,
            n_shared_experts=1,
            n_group=2,
            topk_group=1,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            first_k_dense_replace=1,
            vocab_size=129280,
        )

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)

        # Use DeepSeek-V3 tokenizer since V3.2's config.json uses old rope_scaling
        # format incompatible with transformers>=5.0; tokenizer is identical.
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
