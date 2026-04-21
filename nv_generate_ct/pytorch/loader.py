# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NV-Generate-CT (nvidia/NV-Generate-CT) model loader implementation.

NV-Generate-CT is a 3D latent diffusion model for generating synthetic CT
images with optional anatomical annotations (MAISI v2 from NVIDIA). The
published checkpoints include an AutoencoderKL (AutoencoderKlMaisi) in
image space and a DiffusionModelUNetMaisi operating in the VAE latent
space. This loader targets the latent-space diffusion UNet, which is the
primary generative component invoked during inference.
"""
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

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


REPO_ID = "nvidia/NV-Generate-CT"


class ModelVariant(StrEnum):
    """Available NV-Generate-CT model variants."""

    NV_GENERATE_CT = "NV-Generate-CT"


class ModelLoader(ForgeModel):
    """NV-Generate-CT diffusion UNet loader implementation."""

    _VARIANTS = {
        ModelVariant.NV_GENERATE_CT: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NV_GENERATE_CT

    _WEIGHTS_FILENAME = "models/diff_unet_3d_ddpm-ct.pt"

    # Diffusion UNet architecture (from configs/config_network_ddpm.json).
    _SPATIAL_DIMS = 3
    _IN_CHANNELS = 4
    _OUT_CHANNELS = 4
    _NUM_CHANNELS = (64, 128, 256, 512)
    _ATTENTION_LEVELS = (False, False, True, True)
    _NUM_HEAD_CHANNELS = (0, 0, 32, 32)
    _NUM_RES_BLOCKS = 2

    # Latent spatial resolution corresponds to an 8x downscale from the
    # nominal 512x512x512 CT input supported by the autoencoder.
    _LATENT_SPATIAL = (64, 64, 64)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NV-Generate-CT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the NV-Generate-CT diffusion UNet.

        Downloads the DDPM diffusion UNet checkpoint from HuggingFace Hub
        and loads it into a MONAI ``DiffusionModelUNetMaisi`` instance.

        Returns:
            torch.nn.Module: The loaded diffusion UNet.
        """
        from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
            DiffusionModelUNetMaisi,
        )

        model = DiffusionModelUNetMaisi(
            spatial_dims=self._SPATIAL_DIMS,
            in_channels=self._IN_CHANNELS,
            out_channels=self._OUT_CHANNELS,
            num_channels=list(self._NUM_CHANNELS),
            attention_levels=list(self._ATTENTION_LEVELS),
            num_head_channels=list(self._NUM_HEAD_CHANNELS),
            num_res_blocks=self._NUM_RES_BLOCKS,
            use_flash_attention=False,
            include_top_region_index_input=True,
            include_bottom_region_index_input=True,
            include_spacing_input=True,
        )

        weights_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self._WEIGHTS_FILENAME,
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "unet_state_dict" in state_dict:
            state_dict = state_dict["unet_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the diffusion UNet.

        The MAISI diffusion UNet operates on VAE latents with additional
        scalar conditioning tensors for the top/bottom body regions and
        voxel spacing.

        Returns:
            dict: Keyword arguments for the diffusion UNet forward pass.
        """
        dtype = dtype_override or torch.float32
        latent = torch.randn(
            batch_size, self._IN_CHANNELS, *self._LATENT_SPATIAL, dtype=dtype
        )
        timesteps = torch.tensor([500] * batch_size, dtype=torch.long)

        # Body region one-hot indicators (head/chest/abdomen/lower).
        top_region_index = torch.zeros(batch_size, 4, dtype=dtype)
        top_region_index[:, 0] = 1.0
        bottom_region_index = torch.zeros(batch_size, 4, dtype=dtype)
        bottom_region_index[:, -1] = 1.0

        # Voxel spacing in mm (x, y, z).
        spacing = torch.tensor([[1.5, 1.5, 1.5]] * batch_size, dtype=dtype)

        return {
            "x": latent,
            "timesteps": timesteps,
            "top_region_index_tensor": top_region_index,
            "bottom_region_index_tensor": bottom_region_index,
            "spacing_tensor": spacing,
        }
