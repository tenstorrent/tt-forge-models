# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HeartCodec neural music codec model loader implementation.
"""

import torch
from heartlib.heartcodec.modeling_heartcodec import HeartCodec
from typing import Optional

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
    """Available HeartCodec model variants."""

    HEARTCODEC_OSS_20260123 = "HeartCodec-oss-20260123"


class ModelLoader(ForgeModel):
    """HeartCodec neural music codec model loader implementation."""

    _VARIANTS = {
        ModelVariant.HEARTCODEC_OSS_20260123: ModelConfig(
            pretrained_model_name="HeartMuLa/HeartCodec-oss-20260123",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HEARTCODEC_OSS_20260123

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="HeartCodec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the HeartCodec model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = HeartCodec.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample code inputs for the HeartCodec model.

        HeartCodec's detokenize method expects a 2D tensor of discrete codes
        with shape (num_quantizers, time_steps). Codes are integer indices in
        the range [0, codebook_size).
        """
        # Config: num_quantizers=8, codebook_size=8192, frame rate ~12.5 Hz.
        num_quantizers = 8
        codebook_size = 8192
        num_frames = 25

        codes = torch.randint(
            low=0,
            high=codebook_size,
            size=(num_quantizers, num_frames),
            dtype=torch.long,
        )

        return codes
