# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote Segmentation 3.0 model loader implementation.

Loads the pyannote/segmentation-3.0 speaker segmentation model via
pyannote.audio. The onnx-community/pyannote-segmentation-3.0 HuggingFace
repo hosts only ONNX weights, so we use the upstream PyTorch checkpoint
instead. For gated-model access, set HF_TOKEN; with TT_RANDOM_WEIGHTS=1 we
build the PyanNet architecture from hyperparameters directly.
"""

import os
from typing import Optional

import torch

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
    """Available Pyannote Segmentation 3.0 model variants."""

    SEGMENTATION_3_0 = "Segmentation_3_0"


class ModelLoader(ForgeModel):
    """Pyannote Segmentation 3.0 model loader implementation."""

    _VARIANTS = {
        ModelVariant.SEGMENTATION_3_0: ModelConfig(
            pretrained_model_name="pyannote/segmentation-3.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEGMENTATION_3_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PyannoteSegmentation3_0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Pyannote Segmentation 3.0 model.

        In random-weights mode (TT_RANDOM_WEIGHTS=1), builds the PyanNet
        architecture directly from hyperparameters to avoid downloading the
        gated checkpoint. Otherwise downloads from HuggingFace, which requires
        a token with access to the gated repo (HF_TOKEN env var).
        """
        if os.environ.get("TT_RANDOM_WEIGHTS", "") == "1":
            self.model = self._build_random_model()
        else:
            from pyannote.audio import Model

            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override

            token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
            if token:
                model_kwargs["use_auth_token"] = token
            model_kwargs |= kwargs

            self.model = Model.from_pretrained(
                self._variant_config.pretrained_model_name, **model_kwargs
            )

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def _build_random_model(self):
        """Build a random-weight PyanNet matching Segmentation 3.0.

        Segmentation 3.0 uses the powerset formulation over 3 speakers with
        a maximum of 2 active per frame (7 classes).
        """
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        from pyannote.audio.models.segmentation import PyanNet

        specs = Specifications(
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=10.0,
            classes=["spk1", "spk2", "spk3"],
            powerset_max_classes=2,
            permutation_invariant=True,
            warm_up=(0.0, 0.0),
        )

        model = PyanNet()
        model.specifications = specs
        model.build()
        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample audio inputs for the segmentation model.

        Returns a 10-second mono audio waveform at 16kHz with shape
        (batch_size, num_channels, num_samples) = (1, 1, 160000).
        """
        dtype = dtype_override or torch.float32
        # 10 seconds of mono audio at 16kHz
        waveform = torch.randn(1, 1, 160000, dtype=dtype)
        return [waveform]
