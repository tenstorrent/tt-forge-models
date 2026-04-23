# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AWPortrait-Z (Shakker-Labs/AWPortrait-Z) model loader implementation.

Portrait-beauty LoRA built on the Tongyi-MAI/Z-Image-Turbo base pipeline,
offering native-noise reduction, relit lighting, and more diverse faces
for text-to-image portrait generation.

Reference: https://huggingface.co/Shakker-Labs/AWPortrait-Z
"""

from typing import Any, Optional

import torch
from diffusers import ZImagePipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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
LORA_REPO = "Shakker-Labs/AWPortrait-Z"
LORA_WEIGHT_NAME = "AWPortrait-Z.safetensors"


class ModelVariant(StrEnum):
    """Available AWPortrait-Z model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """AWPortrait-Z LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    prompt = "a professional portrait photo of a woman in a studio setting"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AWPortrait-Z",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the Z-Image-Turbo pipeline with AWPortrait-Z LoRA weights applied."""
        self._pipe = ZImagePipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        # The AWPortrait-Z LoRA has "diffusion_model." prefixed keys with no alpha
        # keys. diffusers' _convert_non_diffusers_z_image_lora_to_diffusers crashes
        # when alpha keys are missing. Pre-process the state dict to bypass the
        # conversion: strip "diffusion_model." and add "transformer." prefix.
        lora_path = hf_hub_download(repo_id=LORA_REPO, filename=LORA_WEIGHT_NAME)
        lora_state_dict = load_file(lora_path)
        lora_state_dict = {
            "transformer." + k.removeprefix("diffusion_model."): v
            for k, v in lora_state_dict.items()
        }
        self._pipe.load_lora_weights(lora_state_dict)
        self._pipe.fuse_lora()
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the DiT transformer with AWPortrait-Z LoRA weights fused.

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

        if self._pipe is None:
            self._load_pipeline(dtype)

        prompt_embeds, _ = self._pipe.encode_prompt(
            prompt=self.prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        num_channels_latents = self._pipe.transformer.in_channels
        vae_scale = self._pipe.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
