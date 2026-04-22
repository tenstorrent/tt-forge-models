# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
strangerzonehf/Flux-Ultimate-LoRA-Collection model loader implementation.

Loads the FLUX.1-dev transformer directly (without the gated pipeline) and
applies a style LoRA from strangerzonehf/Flux-Ultimate-LoRA-Collection for
stylized text-to-image generation.

Repository: https://huggingface.co/strangerzonehf/Flux-Ultimate-LoRA-Collection
"""

from typing import Any, Optional

import torch
from diffusers import FluxTransformer2DModel
from diffusers.loaders.lora_conversion_utils import (
    _convert_kohya_flux_lora_to_diffusers,
)
from huggingface_hub import hf_hub_download
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file

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

LORA_REPO = "strangerzonehf/Flux-Ultimate-LoRA-Collection"

LORA_ANIMEO = "Animeo.safetensors"
LORA_3D_REALISM = "3D-Realism.safetensors"
LORA_CASTOR_3D_PORTRAIT = "Castor-3D-Portrait-Flux-LoRA.safetensors"

FLUX_DEV_CONFIG = {
    "patch_size": 1,
    "in_channels": 64,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": True,
}


class ModelVariant(StrEnum):
    """Available Flux-Ultimate-LoRA-Collection style variants."""

    ANIMEO = "Animeo"
    REALISM_3D = "3D-Realism"
    CASTOR_3D_PORTRAIT = "Castor-3D-Portrait"


_LORA_FILES = {
    ModelVariant.ANIMEO: LORA_ANIMEO,
    ModelVariant.REALISM_3D: LORA_3D_REALISM,
    ModelVariant.CASTOR_3D_PORTRAIT: LORA_CASTOR_3D_PORTRAIT,
}


class ModelLoader(ForgeModel):
    """strangerzonehf/Flux-Ultimate-LoRA-Collection model loader."""

    _VARIANTS = {
        ModelVariant.ANIMEO: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
        ModelVariant.REALISM_3D: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
        ModelVariant.CASTOR_3D_PORTRAIT: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANIMEO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flux-Ultimate-LoRA-Collection",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer_with_lora(self, dtype=None):
        """Load FLUX.1-dev transformer and apply the selected LoRA weights."""
        transformer = FluxTransformer2DModel(**FLUX_DEV_CONFIG)

        lora_file = _LORA_FILES[self._variant]
        lora_path = hf_hub_download(
            repo_id=LORA_REPO,
            filename=lora_file,
        )
        raw_state_dict = load_file(lora_path)
        converted = _convert_kohya_flux_lora_to_diffusers(raw_state_dict)

        lora_weights = {}
        target_modules = set()
        rank = None
        for k, v in converted.items():
            key = (
                k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
            )
            if ".lora_A." in key or ".lora_B." in key:
                lora_weights[key] = v
            if ".lora_A." in key:
                module = key.split(".lora_A.")[0]
                target_modules.add(module)
                if rank is None:
                    rank = v.shape[0]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=list(target_modules),
        )
        transformer.add_adapter(lora_config, adapter_name="default")
        set_peft_model_state_dict(transformer, lora_weights, adapter_name="default")

        if dtype is not None:
            transformer = transformer.to(dtype)

        transformer.eval()
        self.transformer = transformer
        return self.transformer

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load and return the FLUX transformer with the selected LoRA weights applied."""
        if self.transformer is None:
            self._load_transformer_with_lora(dtype=dtype_override)

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs) -> Any:
        """Load and return sample inputs for the FLUX transformer."""
        if self.transformer is None:
            self._load_transformer_with_lora(dtype=dtype_override)

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

        text_ids = torch.zeros(max_sequence_length, 3).to(dtype=dtype)

        guidance = torch.tensor([3.5] * batch_size, dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
