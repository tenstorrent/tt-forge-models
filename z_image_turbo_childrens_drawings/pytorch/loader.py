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

import os
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image

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

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Z-Image-Turbo base pipeline and apply the Children's Drawings LoRA.

        Returns:
            AutoPipelineForText2Image: The pipeline with LoRA adapter applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        if os.environ.get("TT_COMPILE_ONLY_SYSTEM_DESC") or os.environ.get(
            "TT_RANDOM_WEIGHTS"
        ):
            self.pipeline = self._load_with_random_weights(dtype)
        else:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                **kwargs,
            )
            self.pipeline.load_lora_weights(
                ADAPTER_REPO_ID,
                weight_name="z_image_turbo_childrens_drawings.safetensors",
            )
        return self.pipeline

    def _load_with_random_weights(self, dtype):
        """Build the pipeline from configs with random weights for compile-only testing."""
        from diffusers import (
            ZImagePipeline,
            ZImageTransformer2DModel,
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import AutoConfig, Qwen3Model, Qwen2Tokenizer

        repo_id = self._variant_config.pretrained_model_name

        transformer_config = ZImageTransformer2DModel.load_config(
            repo_id, subfolder="transformer"
        )
        transformer = ZImageTransformer2DModel.from_config(transformer_config)
        if dtype != torch.float32:
            transformer = transformer.to(dtype)

        text_encoder_config = AutoConfig.from_pretrained(
            repo_id, subfolder="text_encoder"
        )
        text_encoder = Qwen3Model(text_encoder_config)
        if dtype != torch.float32:
            text_encoder = text_encoder.to(dtype)

        tokenizer = Qwen2Tokenizer.from_pretrained(repo_id, subfolder="tokenizer")

        vae_config = AutoencoderKL.load_config(repo_id, subfolder="vae")
        vae = AutoencoderKL.from_config(vae_config)
        if dtype != torch.float32:
            vae = vae.to(dtype)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            repo_id, subfolder="scheduler"
        )

        return ZImagePipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "a woman holding a coffee cup, in a beanie, sitting at a cafe"
        ] * batch_size
