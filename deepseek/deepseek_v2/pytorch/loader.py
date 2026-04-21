# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V2 model loader implementation for causal language modeling.

Uses reduced MoE configuration for testing since the full 236B parameter
model is too large to load directly.
"""

from typing import Optional

from transformers import AutoTokenizer, DeepseekV2Config, DeepseekV2ForCausalLM
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Moe

_original_moe_init = DeepseekV2Moe.__init__


def _patched_moe_init(self, config):
    _original_moe_init(self, config)
    self.num_experts = config.n_routed_experts


DeepseekV2Moe.__init__ = _patched_moe_init

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """DeepSeek-V2 model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "deepseek-ai/DeepSeek-V2"
        self.tokenizer = None
        self.text = "The capital of France is"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V2",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        num_layers = self.num_layers if self.num_layers is not None else 2

        config = DeepseekV2Config(
            vocab_size=102400,
            hidden_size=1024,
            intermediate_size=1024 * 4,
            moe_intermediate_size=1536,
            num_hidden_layers=num_layers,
            num_attention_heads=16,
            num_key_value_heads=16,
            n_shared_experts=2,
            n_routed_experts=160,
            num_experts_per_tok=2,
            q_lora_rank=256,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
            max_position_embeddings=163840,
            routed_scaling_factor=16.0,
            scoring_func="softmax",
            topk_method="group_limited_greedy",
            n_group=8,
            topk_group=3,
            norm_topk_prob=False,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            attn_implementation="eager",
        )

        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model = DeepseekV2ForCausalLM(config)
        model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
