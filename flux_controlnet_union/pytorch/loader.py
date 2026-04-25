# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX ControlNet Union model loader implementation
"""

import json
import os
import tempfile
import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
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
    """Available FLUX ControlNet Union model variants."""

    FLUX_1_DEV_CONTROLNET_UNION = "FLUX.1-dev-Controlnet-Union"


class ModelLoader(ForgeModel):
    """FLUX ControlNet Union model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX_1_DEV_CONTROLNET_UNION: ModelConfig(
            pretrained_model_name="InstantX/FLUX.1-dev-Controlnet-Union",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX_1_DEV_CONTROLNET_UNION

    # GGUF source for the FLUX.1-dev transformer (non-gated, same architecture)
    _GGUF_REPO_ID = "calcuis/flux1-gguf"
    _GGUF_FILE = "flux1-dev-q4_k_s.gguf"

    # FLUX.1-dev transformer architecture config
    _FLUX_DEV_CONFIG = {
        "_class_name": "FluxTransformer2DModel",
        "_diffusers_version": "0.30.0",
        "attention_head_dim": 128,
        "guidance_embeds": True,
        "in_channels": 64,
        "joint_attention_dim": 4096,
        "num_attention_heads": 24,
        "num_layers": 19,
        "num_single_layers": 38,
        "out_channels": 16,
        "patch_size": 2,
        "pooled_projection_dim": 768,
        "timestep_input_dim": 256,
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX ControlNet Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX transformer model.

        Uses a GGUF-quantized FLUX.1-dev transformer from a non-gated source
        since black-forest-labs/FLUX.1-dev requires HuggingFace authentication.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        with tempfile.TemporaryDirectory() as config_dir:
            with open(os.path.join(config_dir, "config.json"), "w") as f:
                json.dump(self._FLUX_DEV_CONFIG, f)

            self.transformer = FluxTransformer2DModel.from_single_file(
                f"https://huggingface.co/{self._GGUF_REPO_ID}/blob/main/{self._GGUF_FILE}",
                config=config_dir,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )

        self.transformer.eval()
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FLUX transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )

        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)
        guidance = torch.tensor([self.guidance_scale], dtype=dtype).expand(batch_size)

        inputs = {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
