# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-2512 FP8 (E4M3FN) model loader implementation for text-to-image generation.

Loads the FP8-quantized Qwen-Image-2512 transformer single-file checkpoint from
BAJOMX/qwen_image_2512_fp8_e4m3fn (a ComfyUI-packaged mirror of
Comfy-Org/Qwen-Image_ComfyUI) using the upstream Qwen/Qwen-Image-2512
transformer config.

Repository:
- https://huggingface.co/BAJOMX/qwen_image_2512_fp8_e4m3fn
"""

import torch
from diffusers import QwenImageTransformer2DModel
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

FP8_URL = (
    "https://huggingface.co/BAJOMX/qwen_image_2512_fp8_e4m3fn/blob/main/"
    "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"
)
UPSTREAM_REPO = "Qwen/Qwen-Image-2512"


class ModelVariant(StrEnum):
    """Available Qwen-Image-2512 FP8 model variants."""

    FP8_E4M3FN = "fp8_e4m3fn"


class ModelLoader(ForgeModel):
    """Qwen-Image-2512 FP8 (E4M3FN) model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.FP8_E4M3FN: ModelConfig(
            pretrained_model_name="BAJOMX/qwen_image_2512_fp8_e4m3fn",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FP8_E4M3FN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen-Image-2512 FP8 E4M3FN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        load_kwargs = {
            "config": UPSTREAM_REPO,
            "subfolder": "transformer",
        }
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = QwenImageTransformer2DModel.from_single_file(
            FP8_URL,
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Image dimensions
        height = 128
        width = 128
        patch_size = config.patch_size
        in_channels = config.in_channels

        # Compute latent sequence length: (H / patch) * (W / patch)
        h_patches = height // patch_size
        w_patches = width // patch_size
        image_seq_len = h_patches * w_patches

        # Hidden states: (batch, image_seq_len, in_channels)
        hidden_states = torch.randn(batch_size, image_seq_len, in_channels, dtype=dtype)

        # Encoder hidden states (text embeddings): (batch, text_seq_len, joint_attention_dim)
        text_seq_len = 128
        joint_attention_dim = config.joint_attention_dim
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        # Encoder hidden states mask: (batch, text_seq_len)
        encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        # Image shapes for RoPE: list of (frames, height, width)
        img_shapes = [(1, h_patches, w_patches)] * batch_size

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }

        return inputs
