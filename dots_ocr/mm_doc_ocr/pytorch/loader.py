# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr model loader - full end-to-end document-OCR forward.

rednote-hilab/dots.ocr is a vision-language document-OCR model: a NaViT-style
vision tower (``DotsVisionTransformer``) whose image embeddings are scattered
into the Qwen2 decoder's input embeddings (``DotsOCRForCausalLM``, a subclass of
``Qwen2ForCausalLM``). This loader drives the complete image+text forward used
for document parsing.
"""
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
from ...common import (
    DOTS_OCR_MODEL,
    SAMPLE_PROMPT,
    load_full_model,
    load_processor,
    build_multimodal_inputs,
)


class ModelVariant(StrEnum):
    """Available dots.ocr variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Loader for the full dots.ocr document-OCR pipeline (vision + decoder)."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name=DOTS_OCR_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_prompt = SAMPLE_PROMPT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        if self.processor is None:
            self.processor = load_processor()
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the full DotsOCRForCausalLM model."""
        return load_full_model(dtype_override=dtype_override)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build full image+text OCR inputs."""
        self._load_processor()
        return build_multimodal_inputs(
            self.processor, prompt=self.sample_prompt, dtype_override=dtype_override
        )

    def decode_output(self, outputs, inputs=None):
        """Decode the next-token prediction into a token string."""
        self._load_processor()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_id = logits[0, -1].argmax(-1).item()
        return self.processor.tokenizer.decode([next_id])
