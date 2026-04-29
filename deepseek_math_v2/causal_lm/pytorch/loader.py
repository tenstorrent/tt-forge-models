# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-Math-V2 model loader implementation for causal language modeling.

Built on DeepSeek-V3.2-Exp-Base (671B parameters). The full model exceeds
single-device DRAM capacity.
"""

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# deepseek_v32 (DeepSeek-V3.2 Exp) is not in any released transformers version.
# Map it to the V3 architecture, which is structurally compatible minus the
# V3.2 indexer attention head (index_n_heads, index_topk, index_head_dim).


class _DeepseekV32Config(DeepseekV3Config):
    model_type = "deepseek_v32"


class _DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    config_class = _DeepseekV32Config


try:
    AutoConfig.register("deepseek_v32", _DeepseekV32Config)
except ValueError:
    pass
try:
    AutoModelForCausalLM.register(_DeepseekV32Config, _DeepseekV32ForCausalLM)
except ValueError:
    pass


class ModelVariant(StrEnum):
    """Available DeepSeek-Math-V2 model variants for causal language modeling."""

    DEEPSEEK_MATH_V2 = "DeepSeek-Math-V2"


class ModelLoader(ForgeModel):
    """DeepSeek-Math-V2 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_MATH_V2: None,
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_MATH_V2

    sample_text = (
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "What is the integral of x^2 from 0 to 2?"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = "deepseek-ai/DeepSeek-Math-V2"
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DeepSeek-Math-V2",
            variant=variant,
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

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
