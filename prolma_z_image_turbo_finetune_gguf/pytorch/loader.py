# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo Finetune GGUF model loader implementation.

Loads quantized GGUF variants of Z-Image-Turbo community finetunes hosted at
prolma/Z-Image-Turbo-Finetune-GGUFs. The DiT transformer is loaded via the
GGUF single-file API and combined with the upstream Tongyi-MAI/Z-Image-Turbo
pipeline for text-to-image generation.
"""

from typing import Any, Optional

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

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

GGUF_REPO_ID = "prolma/Z-Image-Turbo-Finetune-GGUFs"
PIPELINE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"


class ModelVariant(StrEnum):
    """Available Z-Image-Turbo finetune GGUF model variants."""

    DARKBEASTZ_V6_BLITZ_VII6_Q8_0 = "darkbeastz6Blitz.Vii6-Q8_0"
    MOODY_PORN_MIX_ZITV9_Q4_K_M = "moodyPornMix_zitV9-Q4_K_M"
    MOODY_PORN_MIX_ZITV9_Q6_K = "moodyPornMix_zitV9-Q6_K"
    MOODY_PORN_MIX_ZITV9_Q8_0 = "moodyPornMix_zitV9-Q8_0"
    UNSTABLE_BASTARD_ZBASTARDV20_Q8_0 = "unstablebastard_zBastardV20-Q8_0"


class ModelLoader(ForgeModel):
    """Z-Image-Turbo Finetune GGUF model loader."""

    _VARIANTS = {
        ModelVariant.DARKBEASTZ_V6_BLITZ_VII6_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
        ModelVariant.MOODY_PORN_MIX_ZITV9_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
        ModelVariant.MOODY_PORN_MIX_ZITV9_Q6_K: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
        ModelVariant.MOODY_PORN_MIX_ZITV9_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
        ModelVariant.UNSTABLE_BASTARD_ZBASTARDV20_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MOODY_PORN_MIX_ZITV9_Q4_K_M

    GGUF_FILES = {
        ModelVariant.DARKBEASTZ_V6_BLITZ_VII6_Q8_0: "darkbeastz6Blitz.Vii6-Q8_0.gguf",
        ModelVariant.MOODY_PORN_MIX_ZITV9_Q4_K_M: "moodyPornMix_zitV9-Q4_K_M.gguf",
        ModelVariant.MOODY_PORN_MIX_ZITV9_Q6_K: "moodyPornMix_zitV9-Q6_K.gguf",
        ModelVariant.MOODY_PORN_MIX_ZITV9_Q8_0: "moodyPornMix_zitV9-Q8_0.gguf",
        ModelVariant.UNSTABLE_BASTARD_ZBASTARDV20_Q8_0: "unstablebastard_zBastardV20-Q8_0.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_TURBO_FINETUNE_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(self, dtype: torch.dtype = torch.bfloat16):
        """Load the GGUF-quantized transformer."""
        gguf_file = self.GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=GGUF_REPO_ID, filename=gguf_file)
        transformer = ZImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        return transformer

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the Z-Image-Turbo pipeline with GGUF transformer."""
        transformer = self._load_transformer(dtype)
        self._pipe = ZImagePipeline.from_pretrained(
            PIPELINE_REPO_ID,
            transformer=transformer,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the GGUF-quantized DiT transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the GGUF transformer."""
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
