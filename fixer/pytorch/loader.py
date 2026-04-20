# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Fixer model loader implementation.

Fixer is a single-step image diffusion model (V2 of nvidia/difix) that
enhances rendered novel views by removing artifacts from 3D reconstructions
(NeRF/3DGS). It uses a Linear-attention Diffusion Transformer backbone with
a Deep Compression Autoencoder (DC-AE), built on top of Cosmos-Predict-0.6B.

Reference: https://huggingface.co/nvidia/Fixer
Upstream:  https://github.com/nv-tlabs/Fixer

Available variants:
- BASE: nvidia/Fixer (576x1024 image-to-image enhancement)
"""

from typing import Optional

import torch
from huggingface_hub import hf_hub_download

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

# Input image dimensions expected by the model
IMAGE_HEIGHT = 576
IMAGE_WIDTH = 1024

# Pickled checkpoint path within the Hugging Face repository
CHECKPOINT_FILENAME = "pretrained/pretrained_fixer.pkl"


class ModelVariant(StrEnum):
    """Available Fixer model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """NVIDIA Fixer model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="nvidia/Fixer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Fixer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Fixer diffusion model.

        Downloads the pickled Fixer checkpoint from Hugging Face and loads it
        via ``torch.load``. The checkpoint is a pickled PyTorch module bundling
        the Linear-DiT denoiser and DC-AE autoencoder.
        """
        model_name = self._variant_config.pretrained_model_name

        ckpt_path = hf_hub_download(repo_id=model_name, filename=CHECKPOINT_FILENAME)
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Fixer model.

        Returns:
            torch.Tensor: RGB image tensor of shape (B, 3, 576, 1024).
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        inputs = torch.rand(
            batch_size,
            3,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            dtype=dtype,
        )

        return inputs
