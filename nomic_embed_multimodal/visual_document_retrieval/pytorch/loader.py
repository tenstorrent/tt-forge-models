# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nomic Embed Multimodal model loader implementation for visual document retrieval.
"""
import torch
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
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
    """Available Nomic Embed Multimodal model variants."""

    NOMIC_EMBED_MULTIMODAL_3B = "nomic-embed-multimodal-3b"


class ModelLoader(ForgeModel):
    """Nomic Embed Multimodal model loader for visual document retrieval."""

    _VARIANTS = {
        ModelVariant.NOMIC_EMBED_MULTIMODAL_3B: ModelConfig(
            pretrained_model_name="nomic-ai/nomic-embed-multimodal-3b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOMIC_EMBED_MULTIMODAL_3B

    sample_queries = [
        "What is the revenue for 2024?",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Nomic Embed Multimodal",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = BiQwen2_5_Processor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = BiQwen2_5.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        batch_queries = self.processor.process_queries(self.sample_queries)

        return batch_queries

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs
