# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DDPM (Denoising Diffusion Probabilistic Model) loader implementation
"""

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
import json
import torch
from diffusers import UNet2DModel
from huggingface_hub import hf_hub_download
from typing import Optional


class ModelVariant(StrEnum):
    """Available DDPM model variants."""

    CELEBAHQ_256 = "google/ddpm-celebahq-256"
    DUMMY = "diffusers/ddpm_dummy"


class ModelLoader(ForgeModel):
    """DDPM model loader implementation."""

    _VARIANTS = {
        ModelVariant.CELEBAHQ_256: ModelConfig(
            pretrained_model_name="google/ddpm-celebahq-256",
        ),
        ModelVariant.DUMMY: ModelConfig(
            pretrained_model_name="diffusers/ddpm_dummy",
        ),
    }

    # UNet2DModel constructor keys that map directly from the ddpm_dummy config.json.
    # The dummy repo ships old-style weights (diffusion_model.pt) that modern
    # diffusers cannot load via from_pretrained, so we construct the architecture
    # from its config with random weights — adequate for compilation coverage.
    _DUMMY_CONFIG_KEYS = (
        "sample_size",
        "in_channels",
        "out_channels",
        "down_block_types",
        "up_block_types",
        "block_out_channels",
        "downsample_padding",
        "flip_sin_to_cos",
        "dropout",
    )

    DEFAULT_VARIANT = ModelVariant.CELEBAHQ_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DDPM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DDPM UNet2D model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DModel: The pre-trained unconditional UNet model.
        """
        if self._variant == ModelVariant.DUMMY:
            config_path = hf_hub_download(
                self._variant_config.pretrained_model_name, filename="config.json"
            )
            with open(config_path) as f:
                raw_config = json.load(f)
            filtered_config = {
                k: raw_config[k] for k in self._DUMMY_CONFIG_KEYS if k in raw_config
            }
            filtered_config["layers_per_block"] = raw_config.get("num_res_blocks", 1)
            filtered_config["freq_shift"] = raw_config.get("downscale_freq_shift", 0)
            model = UNet2DModel(**filtered_config)
        else:
            model = UNet2DModel.from_pretrained(
                self._variant_config.pretrained_model_name,
                **kwargs,
            )
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DDPM model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample and timestep inputs.
        """
        sample_size = 32 if self._variant == ModelVariant.DUMMY else 256
        sample = torch.randn(batch_size, 3, sample_size, sample_size)
        timestep = torch.tensor([0])

        if dtype_override is not None:
            sample = sample.to(dtype_override)

        return {
            "sample": sample,
            "timestep": timestep,
        }
