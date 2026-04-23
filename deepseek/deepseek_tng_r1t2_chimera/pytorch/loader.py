# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek TNG R1T2 Chimera model loader implementation for causal language modeling.

This is an Assembly of Experts merge of DeepSeek-R1, DeepSeek-R1-0528, and
DeepSeek-V3-0324. Uses reduced MoE configuration for testing since the full
671B parameter model is too large to load directly.
"""

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """DeepSeek TNG R1T2 Chimera model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "tngtech/DeepSeek-TNG-R1T2-Chimera"
        self.tokenizer = None
        self.text = "Please reason step by step. What is 25 multiplied by 16?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-TNG-R1T2-Chimera",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # tngtech repo has auto_map pointing to modeling_deepseek.py which doesn't
        # exist there; remove it so from_config uses the registered deepseek class
        if hasattr(config, "auto_map"):
            del config.auto_map

        # Custom configuration_deepseek.py uses different attribute names than the
        # registered DeepseekV3Config in transformers:
        # - qk_head_dim is a property (qk_nope_head_dim + qk_rope_head_dim)
        # - n_routed_experts maps to num_local_experts
        if not hasattr(config, "qk_head_dim"):
            config.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        if not hasattr(config, "num_local_experts"):
            config.num_local_experts = config.n_routed_experts

        # Reduce model dimensions for testing
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        else:
            config.num_hidden_layers = 6
        config.num_attention_heads = 16
        config.hidden_size = 1024
        config.num_key_value_heads = 16
        config.intermediate_size = 1024 * 4
        config.num_experts_per_tok = 2
        config.q_lora_rank = 256
        config.use_flash_attention = False

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
