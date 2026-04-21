# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.2 AWQ model loader implementation for causal language modeling.

Uses reduced MoE configuration for testing since the full 685B parameter
AWQ-quantized model is too large to load directly.
"""

import json
from typing import Optional

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """DeepSeek V3.2 AWQ model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "QuantTrio/DeepSeek-V3.2-AWQ"
        self.tokenizer = None
        self.text = "What is machine learning?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V3.2-AWQ",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # Load raw config and override model_type for transformers compatibility
        config_path = hf_hub_download(self.model_name, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        config_dict["model_type"] = "deepseek_v3"
        config = DeepseekV3Config(**config_dict)

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

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = DeepseekV3ForCausalLM._from_config(config, **model_kwargs)

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
