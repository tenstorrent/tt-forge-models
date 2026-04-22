# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pornmaster v1 Z-Image-Turbo LoRA (RomixERR/Pornmaster_v1-Z-Images-Turbo) model
loader implementation.

Loads the Tongyi-MAI/Z-Image-Turbo base pipeline and applies the
RomixERR/Pornmaster_v1-Z-Images-Turbo LoRA adapter weights for text-to-image
generation. The repository ships two training checkpoints that are exposed as
variants.

Available variants:
- V1_000043500: Pornmaster_v1_000043500.safetensors LoRA weights
- V1_000044700: Pornmaster_v1_000044700.safetensors LoRA weights
"""

from typing import Any, Optional

import torch
from diffusers import ZImagePipeline

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

BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
LORA_REPO = "RomixERR/Pornmaster_v1-Z-Images-Turbo"


class ModelVariant(StrEnum):
    """Available Pornmaster v1 Z-Image-Turbo LoRA model variants."""

    V1_000043500 = "Pornmaster_v1_000043500"
    V1_000044700 = "Pornmaster_v1_000044700"


_LORA_FILES = {
    ModelVariant.V1_000043500: "Pornmaster_v1_000043500.safetensors",
    ModelVariant.V1_000044700: "Pornmaster_v1_000044700.safetensors",
}


class ModelLoader(ForgeModel):
    """Pornmaster v1 Z-Image-Turbo LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.V1_000043500: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
        ModelVariant.V1_000044700: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_000044700

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe: Optional[ZImagePipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pornmaster v1 Z-Image-Turbo LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_lora_state_dict(self, lora_file: str) -> dict:
        """Load LoRA state dict, injecting default alpha keys (alpha=rank) when absent.

        diffusers 0.37.1's _convert_non_diffusers_z_image_lora_to_diffusers requires
        alpha keys alongside lora_A/lora_B keys, but this LoRA file was saved without
        them.  alpha=rank gives scale=1.0 so the weights are applied as-is.
        """
        import safetensors.torch as st
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(repo_id=LORA_REPO, filename=lora_file)
        state_dict = st.load_file(local_path)

        # Strip diffusion_model. prefix so diffusers format detection works.
        state_dict = {
            k.removeprefix("diffusion_model."): v for k, v in state_dict.items()
        }

        # Add missing alpha keys with alpha=rank → effective scale = 1.0.
        a_key = ".lora_A.weight"
        for k, v in list(state_dict.items()):
            if k.endswith(a_key):
                alpha_key = k[: -len(a_key)] + ".alpha"
                if alpha_key not in state_dict:
                    state_dict[alpha_key] = torch.tensor(float(v.shape[0]))

        return state_dict

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the Z-Image-Turbo pipeline with Pornmaster v1 LoRA weights applied."""
        self._pipe = ZImagePipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        lora_file = _LORA_FILES[self._variant]
        lora_state_dict = self._load_lora_state_dict(lora_file)
        self._pipe.load_lora_weights(lora_state_dict)
        self._pipe.fuse_lora()

        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the DiT transformer with Pornmaster v1 LoRA weights fused.

        Returns:
            torch.nn.Module: The Z-Image-Turbo DiT transformer with LoRA applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Load and return sample inputs for the transformer.

        Returns:
            list: [latent_input_list, timestep, prompt_embeds]
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "pronmstr. A high quality photograph of a mountain landscape"

        if self._pipe is None:
            self._load_pipeline(dtype)

        # Encode the prompt
        prompt_embeds, _ = self._pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        # Prepare latents
        num_channels_latents = self._pipe.transformer.in_channels
        vae_scale = self._pipe.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        # Prepare timestep
        timestep = torch.tensor([0.5], dtype=dtype)

        # The transformer expects x as list of [1, channels, 1, H, W] tensors
        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
