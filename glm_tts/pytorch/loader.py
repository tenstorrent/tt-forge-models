# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-TTS (zai-org/GLM-TTS) model loader implementation for text-to-speech
tasks.

GLM-TTS is a two-stage zero-shot TTS system whose first stage is a
Llama-based causal language model that converts text into speech-token
sequences, and whose second stage is a flow-matching model that produces
mel-spectrograms consumed by a vocoder. This loader targets the
Llama-based LLM backbone, which is stored in the ``llm/`` subfolder of
the HuggingFace repository alongside the flow-matching and vocoder
components.
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GLM-TTS model variants."""

    GLM_TTS = "GLM-TTS"


class ModelLoader(ForgeModel):
    """GLM-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.GLM_TTS: ModelConfig(
            pretrained_model_name="zai-org/GLM-TTS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_TTS

    _LLM_SUBFOLDER = "llm"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GLM-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {"subfolder": self._LLM_SUBFOLDER}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        # zai-org/GLM-TTS tokenizer references tokenization_chatglm.py which
        # does not exist in the repo. Use dummy token IDs for the LlamaForCausalLM
        # backbone (vocab_size=98304).
        return {"input_ids": torch.zeros((1, 10), dtype=torch.long)}
