# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wonder3D v1.0 model loader implementation.

Wonder3D is a single-image-to-multi-view cross-domain diffusion model that
generates 6 consistent novel views (both RGB and surface normals) from a
single input image, for downstream 3D reconstruction.

The UNet variant uses a custom ``UNetMV2DConditionModel`` that extends the
standard diffusers UNet with multi-view and cross-domain attention. It is
sourced from the Wonder3D GitHub repo.

Available variants:
- UNET: UNetMV2DConditionModel (custom multi-view UNet)
- VAE: AutoencoderKL from the Wonder3D pipeline
"""

import os
import sys
import types
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

REPO_ID = "flamehaze1115/wonder3d-v1.0"
WONDER3D_REPO_PATH = "/tmp/wonder3d_repo"


def _ensure_wonder3d_importable():
    """Ensure the Wonder3D repo is cloned and importable."""
    if "mvdiffusion" not in sys.modules:
        if not os.path.isdir(WONDER3D_REPO_PATH):
            import subprocess

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/xxlong0/Wonder3D.git",
                    WONDER3D_REPO_PATH,
                ]
            )

        if WONDER3D_REPO_PATH not in sys.path:
            sys.path.insert(0, WONDER3D_REPO_PATH)

        mvdiffusion_mod = types.ModuleType("mvdiffusion")
        sys.modules["mvdiffusion"] = mvdiffusion_mod
        mvdiffusion_mod.__path__ = [os.path.join(WONDER3D_REPO_PATH, "mvdiffusion")]


class ModelVariant(StrEnum):
    """Available Wonder3D v1.0 model variants."""

    UNET = "UNet"
    VAE = "VAE"


class ModelLoader(ForgeModel):
    """Wonder3D v1.0 model loader."""

    _VARIANTS = {
        ModelVariant.UNET: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.UNET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Wonder3D-v1.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the model component for the selected variant.

        For UNET: returns the custom UNetMV2DConditionModel.
        For VAE: returns the AutoencoderKL.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self._variant == ModelVariant.VAE:
            from diffusers import AutoencoderKL

            model = AutoencoderKL.from_pretrained(
                REPO_ID, subfolder="vae", torch_dtype=dtype
            )
            model = model.to(dtype)
            model.eval()
            return model

        _ensure_wonder3d_importable()
        from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

        model = UNetMV2DConditionModel.from_pretrained(
            REPO_ID, subfolder="unet", torch_dtype=dtype
        )
        model.eval()
        return model

    def load_inputs(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Prepare sample inputs for the selected variant.

        For UNET: returns (sample, timestep, encoder_hidden_states, class_labels).
        For VAE: returns a latent tensor for decoding.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self._variant == ModelVariant.VAE:
            # AutoencoderKL encode input: [B, in_channels=3, H, W]
            return torch.randn(1, 3, 32, 32, dtype=dtype)

        # UNet inputs. Config: in_channels=8, num_views=6, sample_size=32,
        # cross_attention_dim=768, class_embed_type="projection",
        # projection_class_embeddings_input_dim=10.
        num_views = 6
        sample = torch.randn(num_views, 8, 32, 32, dtype=dtype)
        timestep = torch.tensor([500], dtype=torch.long)
        encoder_hidden_states = torch.randn(num_views, 1, 768, dtype=dtype)
        class_labels = torch.randn(num_views, 10, dtype=dtype)

        return [sample, timestep, encoder_hidden_states, class_labels]
