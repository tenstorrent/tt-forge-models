# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Multilingual CLIP ONNX text encoder loader.

Loads the immich-app/XLM-Roberta-Large-Vit-B-16Plus ONNX export of the
M-CLIP text encoder used for multilingual image-text semantic search in
the Immich photo library application.
"""

import os
from typing import Optional

import onnx
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Multilingual CLIP ONNX model variants."""

    IMMICH_XLM_ROBERTA_LARGE_VIT_B_16PLUS = "immich-XLM-Roberta-Large-Vit-B-16Plus"


class ModelLoader(ForgeModel):
    """Multilingual CLIP ONNX text encoder loader."""

    _VARIANTS = {
        ModelVariant.IMMICH_XLM_ROBERTA_LARGE_VIT_B_16PLUS: ModelConfig(
            pretrained_model_name="immich-app/XLM-Roberta-Large-Vit-B-16Plus",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IMMICH_XLM_ROBERTA_LARGE_VIT_B_16PLUS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._model_dir = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="multilingual-clip",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _ensure_snapshot(self):
        if self._model_dir is None:
            self._model_dir = snapshot_download(
                repo_id=self._variant_config.pretrained_model_name,
                allow_patterns=["textual/*"],
            )
        return self._model_dir

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_dir = self._ensure_snapshot()
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(model_dir, "textual")
            )
        return self.tokenizer

    def load_model(self, **kwargs):
        """Load and return the ONNX text encoder model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        model_dir = self._ensure_snapshot()
        onnx_path = os.path.join(model_dir, "textual", "model.onnx")
        model = onnx.load(onnx_path)
        return model

    def load_inputs(self, batch_size=1, **kwargs):
        """Load and return sample tokenized inputs for the text encoder.

        Args:
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors containing input_ids and attention_mask.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        sentences = [
            "Three blind horses listening to Mozart.",
            "Älgen är skogens konung!",
        ]

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
