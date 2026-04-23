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


def _patch_wonder3d_repo():
    """Patch Wonder3D source files for compatibility with newer diffusers (>=0.28)."""
    unet_cond = os.path.join(
        WONDER3D_REPO_PATH, "mvdiffusion", "models", "unet_mv2d_condition.py"
    )
    unet_blocks = os.path.join(
        WONDER3D_REPO_PATH, "mvdiffusion", "models", "unet_mv2d_blocks.py"
    )
    transformer = os.path.join(
        WONDER3D_REPO_PATH, "mvdiffusion", "models", "transformer_mv2d.py"
    )

    _apply_patches(
        unet_cond,
        [
            # _load_state_dict_into_model moved to model_loading_utils
            (
                "from diffusers.models.modeling_utils import ModelMixin, load_state_dict, _load_state_dict_into_model",
                "from diffusers.models.modeling_utils import ModelMixin, load_state_dict\n"
                "try:\n"
                "    from diffusers.models.model_loading_utils import _load_state_dict_into_model\n"
                "except ImportError:\n"
                "    from diffusers.models.modeling_utils import _load_state_dict_into_model",
            ),
            # unet_2d_blocks moved to unets subpackage
            (
                "from diffusers.models.unet_2d_blocks import (",
                "from diffusers.models.unets.unet_2d_blocks import (",
            ),
            # DIFFUSERS_CACHE and HF_HUB_OFFLINE removed from diffusers.utils
            (
                "from diffusers.utils import (\n"
                "    CONFIG_NAME,\n"
                "    DIFFUSERS_CACHE,\n"
                "    FLAX_WEIGHTS_NAME,\n"
                "    HF_HUB_OFFLINE,\n"
                "    SAFETENSORS_WEIGHTS_NAME,\n"
                "    WEIGHTS_NAME,\n"
                "    _add_variant,\n"
                "    _get_model_file,\n"
                "    deprecate,\n"
                "    is_accelerate_available,\n"
                "    is_safetensors_available,\n"
                "    is_torch_version,\n"
                "    logging,\n"
                ")",
                "from diffusers.utils import (\n"
                "    CONFIG_NAME,\n"
                "    FLAX_WEIGHTS_NAME,\n"
                "    SAFETENSORS_WEIGHTS_NAME,\n"
                "    WEIGHTS_NAME,\n"
                "    _add_variant,\n"
                "    _get_model_file,\n"
                "    deprecate,\n"
                "    is_accelerate_available,\n"
                "    is_safetensors_available,\n"
                "    is_torch_version,\n"
                "    logging,\n"
                ")\n"
                "from huggingface_hub.constants import HF_HUB_CACHE as DIFFUSERS_CACHE, HF_HUB_OFFLINE",
            ),
        ],
    )

    _apply_patches(
        unet_blocks,
        [
            # AdaGroupNorm moved from attention to normalization
            (
                "from diffusers.models.attention import AdaGroupNorm",
                "from diffusers.models.normalization import AdaGroupNorm",
            ),
            # dual_transformer_2d moved to transformers subpackage
            (
                "from diffusers.models.dual_transformer_2d import DualTransformer2DModel",
                "from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel",
            ),
            # unet_2d_blocks moved to unets subpackage (two import lines)
            (
                "from diffusers.models.unet_2d_blocks import DownBlock2D",
                "from diffusers.models.unets.unet_2d_blocks import DownBlock2D",
            ),
            (
                "from diffusers.models.unet_2d_blocks import UpBlock2D",
                "from diffusers.models.unets.unet_2d_blocks import UpBlock2D",
            ),
        ],
    )

    _apply_patches(
        transformer,
        [
            # maybe_allow_in_graph moved to diffusers.utils.torch_utils
            (
                "from diffusers.utils import BaseOutput, deprecate, maybe_allow_in_graph",
                "from diffusers.utils import BaseOutput, deprecate\n"
                "from diffusers.utils.torch_utils import maybe_allow_in_graph",
            ),
        ],
    )


def _apply_patches(filepath, replacements):
    """Apply string replacements to a file; idempotent if already patched."""
    with open(filepath) as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(filepath, "w") as f:
        f.write(content)


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

        _patch_wonder3d_repo()

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
            model.eval()
            return model

        _ensure_wonder3d_importable()
        from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

        model = UNetMV2DConditionModel.from_pretrained(
            REPO_ID, subfolder="unet", torch_dtype=dtype
        )
        model.eval()
        return model

    def load_inputs(self, **kwargs):
        """Prepare sample inputs for the selected variant.

        For UNET: returns (sample, timestep, encoder_hidden_states, class_labels).
        For VAE: returns a latent tensor for decoding.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)

        if self._variant == ModelVariant.VAE:
            # AutoencoderKL decode input: [B, latent_channels, H/8, W/8]
            return torch.randn(1, 4, 32, 32, dtype=dtype)

        # UNet inputs. Config: in_channels=8, num_views=6, sample_size=32,
        # cross_attention_dim=768, class_embed_type="projection",
        # projection_class_embeddings_input_dim=10.
        num_views = 6
        sample = torch.randn(num_views, 8, 32, 32, dtype=dtype)
        timestep = torch.tensor([500], dtype=torch.long)
        encoder_hidden_states = torch.randn(num_views, 1, 768, dtype=dtype)
        class_labels = torch.randn(num_views, 10, dtype=dtype)

        return [sample, timestep, encoder_hidden_states, class_labels]
