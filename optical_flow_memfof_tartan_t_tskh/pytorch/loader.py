# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MEMFOF Tartan-T-TSKH model loader for optical flow estimation.

MEMFOF (Memory-Efficient Multi-Frame Optical Flow) computes backward and
forward optical flow for Full HD video frames. The MEMFOF class is provided
by the memfof package (https://github.com/msu-video-group/memfof).
"""

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
    """Available MEMFOF model variants."""

    TARTAN_T_TSKH = "Tartan-T-TSKH"


class ModelLoader(ForgeModel):
    """MEMFOF Tartan-T-TSKH model loader for optical flow estimation."""

    _VARIANTS = {
        ModelVariant.TARTAN_T_TSKH: ModelConfig(
            pretrained_model_name="egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TARTAN_T_TSKH

    # Full HD input shape: [B, T=3 consecutive frames, C=3 RGB, H, W]
    frame_height = 1080
    frame_width = 1920
    num_frames = 3

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MEMFOF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MEMFOF model instance.

        Requires the memfof package:
            pip install git+https://github.com/msu-video-group/memfof
        """
        from memfof import MEMFOF

        pretrained_model_name = self._variant_config.pretrained_model_name
        model = MEMFOF.from_pretrained(pretrained_model_name, **kwargs)
        model.eval()

        # dtype_override intentionally not applied: the model generates float32
        # coordinate tensors internally, which causes grid_sample to fail when
        # model weights/activations are in bfloat16.

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the MEMFOF model.

        Generates a synthetic triplet of Full HD RGB frames with pixel values
        in the [0, 256) range expected by the model.
        """
        # Always float32: model generates float32 coords internally, so inputs
        # and model must stay in float32 to avoid grid_sample dtype mismatch.
        return torch.randint(
            0,
            256,
            (batch_size, self.num_frames, 3, self.frame_height, self.frame_width),
        ).float()
