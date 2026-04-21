#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pig VAE model loader implementation.

Loads the calcuis/pig-vae GGUF-format VAE models using manual GGUF
dequantization. These are VAE architectures with 16 latent channels
(vs standard SD VAE's 4) distributed in GGUF format.

Available variants:
- SD_VAE_FP16: Stable Diffusion VAE in fp16 GGUF format
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKL
from gguf import GGMLQuantizationType, GGUFReader, dequantize
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

REPO_ID = "calcuis/pig-vae"

SD_VAE_FILENAME = "pig_sd_vae_fp32-f16.gguf"

LATENT_CHANNELS = 16
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


def _load_gguf_as_state_dict(gguf_path, model_state_dict):
    reader = GGUFReader(gguf_path)
    state_dict = {}
    for tensor in reader.tensors:
        data = tensor.data
        if tensor.tensor_type not in (
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
        ):
            data = dequantize(data, tensor.tensor_type)
        weights = torch.from_numpy(data.copy()).float()
        if tensor.name in model_state_dict:
            expected_shape = model_state_dict[tensor.name].shape
            weights = weights.reshape(expected_shape)
        state_dict[tensor.name] = weights
    return state_dict


class ModelVariant(StrEnum):
    """Available Pig VAE model variants."""

    SD_VAE_FP16 = "sd_vae_fp16"


class ModelLoader(ForgeModel):
    """Pig VAE model loader for GGUF-format VAE models."""

    _VARIANTS = {
        ModelVariant.SD_VAE_FP16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SD_VAE_FP16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PIG_VAE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            gguf_path = hf_hub_download(REPO_ID, SD_VAE_FILENAME)

            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4,
                block_out_channels=(128, 256, 512, 512),
                latent_channels=LATENT_CHANNELS,
                layers_per_block=2,
                norm_num_groups=32,
                use_quant_conv=False,
                use_post_quant_conv=False,
            )

            gguf_state_dict = _load_gguf_as_state_dict(gguf_path, vae.state_dict())
            vae.load_state_dict(gguf_state_dict, strict=False)
            vae = vae.to(dtype=dtype)
            vae.eval()
            self._vae = vae
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(
            1,
            IMAGE_CHANNELS,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            dtype=dtype,
        )
