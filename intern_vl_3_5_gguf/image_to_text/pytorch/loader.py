# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternVL3.5 GGUF model loader implementation for image to text.

The GGUF files for InternVL3.5 contain only the Qwen3 text backbone; the
vision encoder weights are absent. This loader targets the text backbone
using AutoModelForCausalLM and uses text-only inputs.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available InternVL3.5 GGUF model variants for image to text."""

    INTERN_VL3_5_4B_Q4_K_M = "4b_q4_k_m"
    INTERN_VL3_5_4B_Q8_0 = "4b_q8_0"
    INTERN_VL3_5_14B_Q4_K_M = "14b_q4_k_m"
    INTERN_VL3_5_14B_Q8_0 = "14b_q8_0"


class ModelLoader(ForgeModel):
    """InternVL3.5 GGUF model loader implementation for image to text tasks.

    Loads the quantized Qwen3 text backbone from the GGUF file. The GGUF
    does not include vision encoder weights, so only the text backbone is tested.
    """

    _VARIANTS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_4B_Q8_0: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q8_0: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
            max_length=128,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab_InternVL3_5-4B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab_InternVL3_5-4B-Q8_0.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab_InternVL3_5-14B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab_InternVL3_5-14B-Q8_0.gguf",
    }

    # Tokenizer sourced from the original HF model (not the GGUF repo)
    _HF_TOKENIZERS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab/InternVL3_5-14B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab/InternVL3_5-14B-HF",
    }

    DEFAULT_VARIANT = ModelVariant.INTERN_VL3_5_4B_Q4_K_M

    sample_text = "What do you see in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InternVL3.5 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._HF_TOKENIZERS[self._variant],
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = gguf_file
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
            padding=True,
        )
        return inputs
