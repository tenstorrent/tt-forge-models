# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo Children's Drawings LoRA (ostris/z_image_turbo_childrens_drawings)
model loader implementation.

Loads the Tongyi-MAI/Z-Image-Turbo base pipeline and applies the
ostris/z_image_turbo_childrens_drawings LoRA adapter to stylize text-to-image
generations in a children's drawing aesthetic.

Available variants:
- Z_IMAGE_TURBO_CHILDRENS_DRAWINGS: Z-Image-Turbo with Children's Drawings LoRA weights applied
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


BASE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"
ADAPTER_REPO_ID = "ostris/z_image_turbo_childrens_drawings"
LORA_WEIGHT_NAME = "z_image_turbo_childrens_drawings.safetensors"


class ModelVariant(StrEnum):
    """Available Z-Image-Turbo Children's Drawings LoRA model variants."""

    Z_IMAGE_TURBO_CHILDRENS_DRAWINGS = "Z_Image_Turbo_Childrens_Drawings"


class ModelLoader(ForgeModel):
    """Z-Image-Turbo Children's Drawings LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE_TURBO_CHILDRENS_DRAWINGS: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_TURBO_CHILDRENS_DRAWINGS

    prompt = "a woman holding a coffee cup, in a beanie, sitting at a cafe"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_Image_Turbo_Childrens_Drawings",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the Z-Image-Turbo pipeline and fuse the Children's Drawings LoRA."""
        self.pipeline = ZImagePipeline.from_pretrained(
            BASE_REPO_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        lora_path = hf_hub_download(
            repo_id=ADAPTER_REPO_ID,
            filename=LORA_WEIGHT_NAME,
        )
        state_dict = load_file(lora_path)
        # diffusers _convert_non_diffusers_z_image_lora_to_diffusers requires
        # alpha keys, but this LoRA file has none; inject rank-valued alphas so
        # the conversion applies a 1.0 scale and leaves weights unchanged.
        a_suffix = ".lora_A.weight"
        alpha_additions = {
            k.replace(a_suffix, ".alpha"): torch.tensor(float(v.shape[0]))
            for k, v in state_dict.items()
            if k.endswith(a_suffix) and k.replace(a_suffix, ".alpha") not in state_dict
        }
        state_dict.update(alpha_additions)
        self.pipeline.load_lora_weights(state_dict)
        self.pipeline.fuse_lora()
        return self.pipeline

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the Z-Image-Turbo transformer with Children's Drawings LoRA fused.

        Returns:
            torch.nn.Module: The Z-Image-Turbo DiT transformer with LoRA applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipeline is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)
        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Load and return sample inputs for the transformer.

        Returns:
            list: [latent_input_list, timestep, prompt_embeds]
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128

        if self.pipeline is None:
            self._load_pipeline(dtype)

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=self.prompt,
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
