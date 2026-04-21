# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CatVTON (zhengchong/CatVTON) virtual try-on model loader implementation.
"""

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
from .src.model_utils import (
    catvton_preprocessing,
    create_dummy_input_image,
    create_dummy_mask_image,
    load_catvton_pipe,
)


BASE_MODEL = "botp/stable-diffusion-v1-5-inpainting"


class ModelVariant(StrEnum):
    """Available CatVTON attention checkpoint variants."""

    MIX_48K_1024 = "mix-48k-1024"


class ModelLoader(ForgeModel):
    """CatVTON virtual try-on model loader implementation."""

    _VARIANTS = {
        ModelVariant.MIX_48K_1024: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIX_48K_1024

    prompt = ""

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CatVTON",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CatVTON virtual try-on pipeline.

        Args:
            dtype_override: Optional torch.dtype to cast the pipeline to.

        Returns:
            StableDiffusionInpaintPipeline: CatVTON-configured inpainting pipeline.
        """
        base_model_name = self._variant_config.pretrained_model_name
        attn_ckpt_version = str(self._variant)

        self.pipeline = load_catvton_pipe(base_model_name, attn_ckpt_version)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return UNet-ready inputs for CatVTON.

        Returns:
            list: [latent_model_input, timestep, prompt_embeds]
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        input_image = create_dummy_input_image()
        mask_image = create_dummy_mask_image()

        (
            latent_model_input,
            timestep,
            prompt_embeds,
        ) = catvton_preprocessing(self.pipeline, self.prompt, input_image, mask_image)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timestep, prompt_embeds]
