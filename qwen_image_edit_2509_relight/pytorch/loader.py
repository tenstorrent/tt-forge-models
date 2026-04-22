# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Image Edit 2509 Relight LoRA model loader implementation.

Loads the dx8152/Qwen-Image-Edit-2509-Relight LoRA adapter on top of the
Qwen/Qwen-Image-Edit-2509 base diffusion pipeline for image relighting.
Returns the inner QwenImageTransformer2DModel with synthetic tensor inputs
so the test harness can compile the transformer directly.
"""

import torch
from diffusers import QwenImageEditPlusPipeline
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
    """Available Qwen Image Edit Relight model variants."""

    RELIGHT = "relight"


class ModelLoader(ForgeModel):
    """Qwen Image Edit 2509 Relight LoRA model loader."""

    _VARIANTS = {
        ModelVariant.RELIGHT: ModelConfig(
            pretrained_model_name="dx8152/Qwen-Image-Edit-2509-Relight",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RELIGHT

    _BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"

    _LORA_WEIGHT_NAMES = {
        ModelVariant.RELIGHT: "Qwen-Edit-Relight.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Qwen Image Edit 2509 Relight",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            self._BASE_MODEL, torch_dtype=dtype, **kwargs
        )
        self.pipe.load_lora_weights(
            self._variant_config.pretrained_model_name,
            weight_name=self._LORA_WEIGHT_NAMES[self._variant],
        )
        self.pipe.transformer.eval()
        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipe is None:
            self.load_model(dtype_override=dtype_override)

        transformer = self.pipe.transformer
        dtype = dtype_override or torch.bfloat16

        # Use a small image to keep memory footprint low.
        height, width = 128, 128
        vae_scale_factor = getattr(self.pipe, "vae_scale_factor", 8)

        # img_shapes: (C, H_latent, W_latent) per image in each batch element.
        # For an edit task the latent_model_input concatenates the noisy latent
        # with the condition latent along the sequence dimension, so we provide
        # two shapes per batch element.
        h_lat = height // vae_scale_factor // 2
        w_lat = width // vae_scale_factor // 2
        img_shapes = [[(1, h_lat, w_lat), (1, h_lat, w_lat)]] * batch_size

        in_channels = transformer.config.in_channels
        # Two images concatenated along sequence dim (noisy + condition).
        seq_len = h_lat * w_lat * 2
        hidden_states = torch.randn(batch_size, seq_len, in_channels, dtype=dtype)

        joint_attention_dim = transformer.config.joint_attention_dim
        text_seq_len = 64
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(
            batch_size, text_seq_len, dtype=torch.bool
        )

        timestep = torch.full((batch_size,), 0.5, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
