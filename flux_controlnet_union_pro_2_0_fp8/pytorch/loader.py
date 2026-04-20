# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-dev ControlNet Union Pro 2.0 FP8 model loader implementation.

Loads an FP8-quantized (E4M3) variant of the Shakker-Labs
FLUX.1-dev-ControlNet-Union-Pro-2.0 ControlNet. The ControlNet is loaded
directly from the HF repo and is paired with synthetic inputs sized to match
its forward signature so no gated base model is required.

Repository: https://huggingface.co/ABDALLALSWAITI/FLUX.1-dev-ControlNet-Union-Pro-2.0-fp8
"""

import torch
from diffusers.models import FluxControlNetModel
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
    """Available FLUX.1-dev ControlNet Union Pro 2.0 FP8 model variants."""

    FLUX_1_DEV_CONTROLNET_UNION_PRO_2_0_FP8 = (
        "FLUX.1-dev-ControlNet-Union-Pro-2.0-fp8"
    )


class ModelLoader(ForgeModel):
    """FLUX.1-dev ControlNet Union Pro 2.0 FP8 model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX_1_DEV_CONTROLNET_UNION_PRO_2_0_FP8: ModelConfig(
            pretrained_model_name="ABDALLALSWAITI/FLUX.1-dev-ControlNet-Union-Pro-2.0-fp8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX_1_DEV_CONTROLNET_UNION_PRO_2_0_FP8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.controlnet = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-dev ControlNet Union Pro 2.0 FP8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FP8-quantized FLUX ControlNet model.

        Args:
            dtype_override: Optional torch.dtype. Defaults to bfloat16 so the
                stored float8_e4m3fn weights are upcast to a compute-capable
                dtype.

        Returns:
            torch.nn.Module: The FLUX ControlNet model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.controlnet = FluxControlNetModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        self.controlnet.eval()
        for param in self.controlnet.parameters():
            param.requires_grad = False

        return self.controlnet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample inputs for the FLUX ControlNet Union Pro 2.0 model.

        Args:
            dtype_override: Optional torch.dtype to override the default input
                dtype of bfloat16.
            batch_size: Batch size (default: 1).

        Returns:
            dict: Input tensors matching FluxControlNetModel.forward.
        """
        if self.controlnet is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.controlnet.config

        # Image dimensions
        height = 128
        width = 128
        vae_scale_factor = 8
        max_sequence_length = 256
        num_channels_latents = config.in_channels // 4

        # Packed latent layout: (B, H*W, C*4)
        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size,
            h_packed * w_packed,
            num_channels_latents * 4,
            dtype=dtype,
        )

        # Pro 2.0 has no input hint block, so controlnet_cond shares the latent shape.
        controlnet_cond = torch.randn_like(latents)

        # Text embeddings
        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        prompt_embeds = torch.randn(
            batch_size,
            max_sequence_length,
            config.joint_attention_dim,
            dtype=dtype,
        )

        # Text IDs (2D per diffusers 0.31+)
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        # Latent image IDs (2D)
        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        guidance = (
            torch.full([batch_size], self.guidance_scale, dtype=dtype)
            if config.guidance_embeds
            else None
        )

        inputs = {
            "hidden_states": latents,
            "controlnet_cond": controlnet_cond,
            "controlnet_mode": None,
            "conditioning_scale": 1.0,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "img_ids": latent_image_ids,
            "txt_ids": text_ids,
            "guidance": guidance,
            "joint_attention_kwargs": {},
        }

        return inputs
