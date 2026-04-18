# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NSFW-MASTER-Z-IMAGE-TURBO LoRA model loader implementation.

Loads the Tongyi-MAI/Z-Image-Turbo base pipeline and applies the
thutes-gbr25/NSFW-MASTER-Z-IMAGE-TURBO LoRA adapter for text-to-image generation.

Available variants:
- NSFW_MASTER_Z_IMAGE_TURBO: Z-Image-Turbo with NSFW-MASTER LoRA weights applied
"""

import re
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
LORA_REPO = "thutes-gbr25/NSFW-MASTER-Z-IMAGE-TURBO"


class ModelVariant(StrEnum):
    """Available NSFW-MASTER-Z-IMAGE-TURBO model variants."""

    NSFW_MASTER_Z_IMAGE_TURBO = "nsfw_master_z_image_turbo"


class ModelLoader(ForgeModel):
    """NSFW-MASTER-Z-IMAGE-TURBO LoRA model loader."""

    _VARIANTS = {
        ModelVariant.NSFW_MASTER_Z_IMAGE_TURBO: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NSFW_MASTER_Z_IMAGE_TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[ZImagePipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NSFW_MASTER_Z_IMAGE_TURBO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _convert_lora_state_dict(state_dict):
        """Convert non-diffusers Z-Image LoRA state dict to diffusers format.

        Works around a diffusers bug where normalize_out_key only handles
        lora_down/lora_up key suffixes but not lora_A/lora_B, causing a
        KeyError when alpha keys get renamed but weight keys do not.
        """
        state_dict = {
            k.removeprefix("diffusion_model."): v for k, v in state_dict.items()
        }

        def normalize_out(k):
            return re.sub(
                r"\.out(?=\.(?:lora_A|lora_B|lora_down|lora_up)\.weight$|\.alpha$)",
                ".to_out.0",
                k,
            )

        state_dict = {normalize_out(k): v for k, v in state_dict.items()}

        converted = {}
        for k in list(state_dict.keys()):
            if not k.endswith(".lora_A.weight"):
                continue
            base = k[: -len(".lora_A.weight")]
            alpha_key = f"{base}.alpha"
            b_key = f"{base}.lora_B.weight"
            down = state_dict[k]
            up = state_dict[b_key]
            rank = down.shape[0]
            alpha = state_dict[alpha_key].item()
            scale = alpha / rank
            converted[f"transformer.{k}"] = down * scale
            converted[f"transformer.{b_key}"] = up
        return converted

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load Z-Image-Turbo pipeline with NSFW-MASTER LoRA weights applied.

        Returns:
            The DiT transformer from the pipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.pipeline = ZImagePipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        lora_path = hf_hub_download(
            LORA_REPO, filename="NSFW_master_ZIT_000008766.safetensors"
        )
        raw_sd = load_file(lora_path)
        converted_sd = self._convert_lora_state_dict(raw_sd)
        self.pipeline.load_lora_weights(converted_sd)

        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the transformer.

        Returns:
            List of [latent_input_list, timestep, prompt_embeds].
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A photo of an astronaut riding a horse on mars"

        if self.pipeline is None:
            self.pipeline = ZImagePipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        num_channels_latents = self.pipeline.transformer.in_channels
        vae_scale = self.pipeline.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
