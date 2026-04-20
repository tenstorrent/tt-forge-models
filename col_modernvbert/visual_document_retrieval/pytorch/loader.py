# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ColModernVBERT model loader implementation for visual document retrieval.
"""
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ColModernVBERT model variants."""

    COLMODERNVBERT = "colmodernvbert"


class ModelLoader(ForgeModel):
    """ColModernVBERT model loader for visual document retrieval."""

    _VARIANTS = {
        ModelVariant.COLMODERNVBERT: ModelConfig(
            pretrained_model_name="ModernVBERT/colmodernvbert",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COLMODERNVBERT

    sample_texts = [
        "What is the revenue for 2024?",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ColModernVBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = ColModernVBertProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ColModernVBert.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        text_inputs = self.processor.process_texts(self.sample_texts)

        return text_inputs

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs
