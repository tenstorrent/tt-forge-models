# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote speaker segmentation model loader implementation.
"""

import os

import torch
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
    """Available Pyannote segmentation model variants."""

    SEGMENTATION = "Segmentation"
    SEGMENTATION_3_0 = "Segmentation_3_0"


class ModelLoader(ForgeModel):
    """Pyannote speaker segmentation model loader implementation."""

    _VARIANTS = {
        ModelVariant.SEGMENTATION: ModelConfig(
            pretrained_model_name="pyannote/segmentation",
        ),
        ModelVariant.SEGMENTATION_3_0: ModelConfig(
            pretrained_model_name="pyannote/segmentation-3.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEGMENTATION

    def __init__(self, variant=None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pyannote",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Pyannote segmentation model.

        Requires a HuggingFace token with access to the gated model.
        Set the HF_TOKEN environment variable or pass token as a kwarg.
        """
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            self._model = self._build_random_model(dtype_override)
        else:
            from pyannote.audio import Model

            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override

            token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
            model_kwargs |= kwargs

            self._model = Model.from_pretrained(
                self._variant_config.pretrained_model_name,
                token=token,
                **model_kwargs,
            )
            self._model.eval()
            if dtype_override is not None:
                self._model.to(dtype_override)
        return self._model

    @staticmethod
    def _build_random_model(dtype_override=None):
        from pyannote.audio.models.segmentation import PyanNet
        from pyannote.audio.core.model import Specifications
        from pyannote.audio.core.task import Problem, Resolution

        model = PyanNet(
            sincnet={"stride": 10},
            lstm={"hidden_size": 128, "num_layers": 4, "bidirectional": True},
            linear={"hidden_size": 128, "num_layers": 2},
        )
        model.specifications = Specifications(
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=10.0,
            classes=["speaker#1", "speaker#2", "speaker#3"],
            powerset_max_classes=2,
            permutation_invariant=True,
        )
        model.build()
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the segmentation model.

        Generates a 10-second mono audio waveform at 16kHz as expected
        by the model: shape (batch_size, num_channels, num_samples) = (1, 1, 160000).
        """
        dtype = dtype_override or torch.float32
        # 10 seconds of mono audio at 16kHz
        waveform = torch.randn(1, 1, 160000, dtype=dtype)
        return [waveform]
