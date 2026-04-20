# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEYOND REALITY Z IMAGE model loader implementation.

Loads a merged fine-tune of Tongyi-MAI/Z-Image-Turbo from the
Nurburgring/BEYOND_REALITY_Z_IMAGE repository. The fine-tuned DiT
transformer weights are downloaded as a single safetensors file and
loaded on top of the base Z-Image-Turbo pipeline.

Available variants:
- BEYOND_REALITY_Z_IMAGE_3_BF16: BEYOND REALITY SUPER Z IMAGE 3.0 (Rebuild) BF16
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

REPO_ID = "Nurburgring/BEYOND_REALITY_Z_IMAGE"
BASE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"


class ModelVariant(StrEnum):
    """Available BEYOND REALITY Z IMAGE model variants."""

    BEYOND_REALITY_Z_IMAGE_3_BF16 = "3.0-BF16"


VARIANT_FILENAMES = {
    ModelVariant.BEYOND_REALITY_Z_IMAGE_3_BF16: (
        "BEYOND REALITY SUPER Z IMAGE 3.0 \u6de1\u5986\u6d53\u62b9 BF16.safetensors"
    ),
}


class ModelLoader(ForgeModel):
    """BEYOND REALITY Z IMAGE model loader."""

    _VARIANTS = {
        ModelVariant.BEYOND_REALITY_Z_IMAGE_3_BF16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BEYOND_REALITY_Z_IMAGE_3_BF16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BEYOND_REALITY_Z_IMAGE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the base Z-Image-Turbo pipeline and swap in BEYOND REALITY weights."""
        self._pipe = ZImagePipeline.from_pretrained(
            BASE_REPO_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        filename = VARIANT_FILENAMES[self._variant]
        ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        state_dict = load_file(ckpt_path)
        self._pipe.transformer.load_state_dict(state_dict)
        self._pipe.transformer.eval()
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the fine-tuned DiT transformer.

        Returns the DiT transformer component from the pipeline with the
        BEYOND REALITY fine-tuned weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the transformer.

        Returns:
            List of [latent_input_list, timestep, prompt_embeds].
        """
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
