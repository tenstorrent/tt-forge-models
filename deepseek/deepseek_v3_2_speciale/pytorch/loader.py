# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.2 Speciale model loader implementation for causal language modeling.
"""

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)

# DeepSeek-V3.2-Speciale uses model_type "deepseek_v32" which is not yet in transformers 5.x;
# register it as an alias of the structurally identical DeepseekV3Config so AutoConfig resolves it.
if "deepseek_v32" not in CONFIG_MAPPING:
    CONFIG_MAPPING.register("deepseek_v32", DeepseekV3Config, exist_ok=True)


class ModelLoader(ForgeModel):
    """DeepSeek V3.2 Speciale model loader for causal language modeling."""

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = "deepseek-ai/DeepSeek-V3.2-Speciale"
        self.tokenizer = None
        self.text = "Please reason step by step. What is 25 multiplied by 16?"

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
        config = AutoConfig.from_pretrained(self.model_name)

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
