# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo LightX2V Quantized model loader implementation.

Loads quantized Z-Image-Turbo diffusion transformer weights from
lightx2v/Z-Image-Turbo-Quantized, swapping them into the base
Tongyi-MAI/Z-Image-Turbo diffusers pipeline.

Available variants:
- FP8_E4M3FN: Scaled FP8 E4M3FN quantized transformer
- INT8: INT8 quantized transformer
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

REPO_ID = "lightx2v/Z-Image-Turbo-Quantized"
BASE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"


class ModelVariant(StrEnum):
    """Available Z-Image-Turbo LightX2V Quantized model variants."""

    FP8_E4M3FN = "FP8_E4M3FN"
    INT8 = "INT8"


VARIANT_FILENAMES = {
    ModelVariant.FP8_E4M3FN: "z_image_turbo_scaled_fp8_e4m3fn.safetensors",
    ModelVariant.INT8: "z_image_turbo_int8.safetensors",
}


class ModelLoader(ForgeModel):
    """Z-Image-Turbo LightX2V Quantized model loader."""

    _VARIANTS = {
        ModelVariant.FP8_E4M3FN: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.INT8: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FP8_E4M3FN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_TURBO_LIGHTX2V_QUANTIZED",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _dequantize_scaled_fp8(state_dict: dict, dtype: torch.dtype) -> dict:
        """Convert scaled-FP8 state dict to dtype by multiplying each weight by its scale."""
        result = {}
        for key, value in state_dict.items():
            if key.endswith("_scale"):
                continue
            scale_key = key + "_scale"
            if scale_key in state_dict:
                result[key] = (value.float() * state_dict[scale_key]).to(dtype)
            else:
                result[key] = value
        return result

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the base Z-Image-Turbo pipeline and swap in quantized weights."""
        self._pipe = ZImagePipeline.from_pretrained(
            BASE_REPO_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        filename = VARIANT_FILENAMES[self._variant]
        ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        state_dict = load_file(ckpt_path)
        if self._variant == ModelVariant.FP8_E4M3FN:
            state_dict = self._dequantize_scaled_fp8(state_dict, dtype)
        self._pipe.transformer.load_state_dict(state_dict)
        self._pipe.transformer.eval()
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the quantized Z-Image-Turbo DiT transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare transformer inputs (latents, timestep, prompt_embeds)."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A photo of an astronaut riding a horse on mars"

        if self._pipe is None:
            self._load_pipeline(dtype)

        prompt_embeds, _ = self._pipe.encode_prompt(
            prompt=prompt,
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
