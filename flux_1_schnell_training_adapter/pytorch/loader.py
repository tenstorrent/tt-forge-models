# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-schnell Training Adapter (ostris/FLUX.1-schnell-training-adapter) model loader.

This is a LoRA de-distillation adapter for the black-forest-labs/FLUX.1-schnell
base model. It is designed to be stacked during LoRA fine-tuning to preserve
step-distillation properties of schnell. The adapter loads the FP8-quantized
FLUX.1-schnell transformer (from the non-gated Kijai/flux-fp8 repo) and applies
the LoRA weights directly to the transformer.

Available variants:
- FLUX_1_SCHNELL_TRAINING_ADAPTER: ostris/FLUX.1-schnell-training-adapter
"""

import json
import os
import tempfile
from typing import Any, Optional

import torch
from diffusers.models import FluxTransformer2DModel

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


ADAPTER_REPO_ID = "ostris/FLUX.1-schnell-training-adapter"
ADAPTER_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

# Non-gated FP8 checkpoint for FLUX.1-schnell (same architecture as the gated base)
_FP8_SCHNELL_URL = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8-e4m3fn.safetensors"

# FLUX transformer architecture config (schnell: no guidance_embeds)
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


class ModelVariant(StrEnum):
    """Available FLUX.1-schnell Training Adapter model variants."""

    FLUX_1_SCHNELL_TRAINING_ADAPTER = "Flux_1_Schnell_Training_Adapter"


class ModelLoader(ForgeModel):
    """FLUX.1-schnell Training Adapter model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX_1_SCHNELL_TRAINING_ADAPTER: ModelConfig(
            pretrained_model_name=ADAPTER_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FLUX_1_SCHNELL_TRAINING_ADAPTER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-schnell-training-adapter",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_local_config_dir(self):
        """Create a temporary directory with the transformer config.json."""
        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(_TRANSFORMER_CONFIG, f)
        return config_dir

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load FP8 FLUX.1-schnell transformer and apply the training LoRA adapter.

        Uses the non-gated Kijai/flux-fp8 checkpoint as the base and applies
        LoRA weights from ostris/FLUX.1-schnell-training-adapter.

        Returns:
            FluxTransformer2DModel: The transformer with LoRA adapter weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        config_dir = self._make_local_config_dir()
        self._transformer = FluxTransformer2DModel.from_single_file(
            _FP8_SCHNELL_URL,
            config=config_dir,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.load_lora_adapter(
            ADAPTER_REPO_ID,
            weight_name=ADAPTER_WEIGHT_NAME,
        )
        return self._transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        """Generate synthetic inputs for the FLUX.1-schnell transformer with LoRA.

        Returns:
            dict: Input tensors for the transformer model.
        """
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        max_sequence_length = 256
        height = 128
        width = 128
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.float32
        num_channels_latents = self._transformer.config.in_channels // 4
        vae_scale_factor = 8

        pooled_projection_dim = self._transformer.config.pooled_projection_dim
        pooled_prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt, pooled_projection_dim, dtype=dtype
        )

        joint_attention_dim = self._transformer.config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt,
            max_sequence_length,
            joint_attention_dim,
            dtype=dtype,
        )

        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))

        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
        )

        latents = torch.randn(shape, dtype=dtype)
        latents = latents.view(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent // 2,
            2,
            width_latent // 2,
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size * num_images_per_prompt,
            (height_latent // 2) * (width_latent // 2),
            num_channels_latents * 4,
        )

        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": None,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
