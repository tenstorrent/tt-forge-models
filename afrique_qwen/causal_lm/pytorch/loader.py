# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AfriqueQwen model loader implementation for causal language modeling.

``israel/AfriqueQwen-14B-Fact-Lora`` is a fully-merged fine-tune of Qwen3-14B
(``Qwen3ForCausalLM`` architecture, standard safetensors weights, no PEFT
adapter). It therefore reuses the existing Qwen 3 causal-LM loader machinery
(model loading, chat-template inputs, sharding, decode) and only overrides the
available variants to point at the merged checkpoint.
"""
from typing import Optional

from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....qwen_3.causal_lm.pytorch.loader import ModelLoader as _Qwen3ModelLoader


class ModelVariant(StrEnum):
    """Available AfriqueQwen model variants for causal language modeling."""

    AFRIQUE_QWEN_14B_FACT = "14b_fact"


class ModelLoader(_Qwen3ModelLoader):
    """AfriqueQwen model loader for causal language modeling tasks.

    Inherits all behaviour from the Qwen 3 causal-LM loader and overrides only
    the variant configuration to point at the merged AfriqueQwen checkpoint.
    """

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.AFRIQUE_QWEN_14B_FACT: LLMModelConfig(
            pretrained_model_name="israel/AfriqueQwen-14B-Fact-Lora",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.AFRIQUE_QWEN_14B_FACT

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="AfriqueQwen 14B Fact",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )
