# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Magi (The Manga Whisperer) model loader implementation for manga panel, character,
and text detection.
"""

from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Magi model variants."""

    MAGI = "magi"


class ModelLoader(ForgeModel):
    """Magi (The Manga Whisperer) model loader for manga detection and transcription."""

    _VARIANTS = {
        ModelVariant.MAGI: ModelConfig(
            pretrained_model_name="ragavsachdeva/magi",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAGI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Magi",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"].convert("L").convert("RGB")
        images = [np.array(image)] * batch_size

        inputs = self.model.processor.preprocess_inputs_for_detection(images)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
