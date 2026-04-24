# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Surya Recognition model loader implementation for OCR text recognition tasks.
"""
import math

import numpy as np
import torch
from PIL import Image
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
    """Available Surya Recognition model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Surya Recognition model loader for OCR text recognition."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="vikp/surya_rec",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # surya 0.17+ model config constants
    _PATCH_SIZE = 14
    _MERGE_SIZE = 2
    _TEMPORAL_PATCH_SIZE = 1
    _IN_CHANNELS = 3
    _IMAGE_WIDTH = 896
    _IMAGE_HEIGHT = 196

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="surya_rec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import importlib
        import os
        import sys

        # Purge any stale surya modules cached from a previous version to ensure
        # we import from the version installed on disk.
        stale = [k for k in list(sys.modules.keys()) if k.split(".")[0] == "surya"]
        for k in stale:
            del sys.modules[k]
        importlib.invalidate_caches()

        from safetensors.torch import load_file
        from surya.common.surya import SuryaModel
        from surya.common.surya.config import SuryaModelConfig
        from surya.common.surya.encoder import SuryaEncoderModel
        from surya.settings import settings

        # Load the encoder directly from the checkpoint to avoid decoder
        # initialization issues (surya 0.17.1 + transformers 5.x incompatibility).
        checkpoint = settings.FOUNDATION_MODEL_CHECKPOINT
        local_path = SuryaModel.get_local_path(checkpoint)

        config = SuryaModelConfig.from_pretrained(local_path)
        config.vision_encoder._attn_implementation = "sdpa"
        config.vision_encoder._attn_implementation_autoset = True

        model = SuryaEncoderModel(config.vision_encoder)

        state_dict = load_file(os.path.join(local_path, "model.safetensors"))
        prefix = "vision_encoder."
        encoder_state_dict = {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        model.load_state_dict(encoder_state_dict, strict=True)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        patch_size = self._PATCH_SIZE
        merge_size = self._MERGE_SIZE
        temporal_patch_size = self._TEMPORAL_PATCH_SIZE
        in_channels = self._IN_CHANNELS
        factor = patch_size * merge_size

        image = Image.new(
            "RGB", (self._IMAGE_WIDTH, self._IMAGE_HEIGHT), color=(255, 255, 255)
        )
        image_np = np.asarray(image, dtype=np.float32)

        height, width = image_np.shape[:2]
        h_bar = math.ceil(height / factor) * factor
        w_bar = math.ceil(width / factor) * factor
        if h_bar != height or w_bar != width:
            import cv2

            image_np = cv2.resize(
                image_np, (w_bar, h_bar), interpolation=cv2.INTER_CUBIC
            )
            height, width = h_bar, w_bar

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_np = (image_np.astype(np.float64) / 255.0).astype(np.float32)
        image_np = (image_np - mean) / std

        img_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
        patches = img_tensor.unsqueeze(0)

        grid_t = patches.shape[0]
        grid_h = height // patch_size
        grid_w = width // patch_size

        patches = patches.reshape(
            grid_t,
            1,
            in_channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            in_channels * temporal_patch_size * patch_size * patch_size,
        )

        image_batch = flatten_patches.unsqueeze(0)
        grid_thw = torch.tensor([[[grid_t, grid_h, grid_w]]], dtype=torch.long)

        if dtype_override is not None:
            image_batch = image_batch.to(dtype_override)

        if batch_size > 1:
            image_batch = image_batch.repeat(batch_size, 1, 1)
            grid_thw = grid_thw.repeat(batch_size, 1, 1)

        return (image_batch, grid_thw)
