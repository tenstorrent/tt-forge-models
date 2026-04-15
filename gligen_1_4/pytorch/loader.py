# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLIGEN 1.4 UNet model loader implementation

Extracts the UNet from the GLIGEN Stable Diffusion pipeline for grounded
text-to-image generation with bounding box control.

Reference: https://huggingface.co/masterful/gligen-1-4-generation-text-box
"""

from typing import Optional

import torch
from diffusers import StableDiffusionGLIGENPipeline
from diffusers.models.attention import GatedSelfAttentionDense
from transformers import CLIPTextModel, CLIPTokenizer

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


class ModelVariant(StrEnum):
    """Available GLIGEN model variants."""

    GENERATION_TEXT_BOX = "generation-text-box"


class ModelLoader(ForgeModel):
    """GLIGEN 1.4 UNet model loader implementation."""

    _VARIANTS = {
        ModelVariant.GENERATION_TEXT_BOX: ModelConfig(
            pretrained_model_name="masterful/gligen-1-4-generation-text-box",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERATION_TEXT_BOX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GLIGEN 1.4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the GLIGEN pipeline and return its UNet.

        Returns:
            UNet2DConditionModel: The UNet from the GLIGEN pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionGLIGENPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        # Enable the GLIGEN fuser modules in the UNet
        for module in pipe.unet.modules():
            if isinstance(module, GatedSelfAttentionDense):
                module.enabled = True

        # Store components needed for input generation
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.scheduler = pipe.scheduler
        self.cross_attention_dim = pipe.unet.config.cross_attention_dim
        self.in_channels = pipe.unet.config.in_channels

        return pipe.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample tensor inputs for the GLIGEN UNet.

        Returns:
            dict: Dictionary with sample, timestep, encoder_hidden_states,
                  and cross_attention_kwargs containing GLIGEN grounding data.
        """
        dtype = dtype_override or torch.bfloat16

        # Encode prompt text
        prompt = [
            "a waterfall and a modern high speed train in a beautiful forest with fall foliage"
        ] * batch_size
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(text_input.input_ids)[0].to(dtype)

        # Generate latent noise input
        height, width = 512, 512
        latents = torch.randn(
            (batch_size, self.in_channels, height // 8, width // 8), dtype=dtype
        )

        self.scheduler.set_timesteps(1)
        latent_model_input = self.scheduler.scale_model_input(
            latents, self.scheduler.timesteps[0]
        )

        # Prepare GLIGEN grounding inputs (boxes, phrase embeddings, masks)
        max_objs = 30
        gligen_phrases = ["a waterfall", "a modern high speed train"]
        gligen_boxes = [
            [0.1387, 0.2051, 0.4277, 0.7090],
            [0.4980, 0.4355, 0.8516, 0.7266],
        ]
        n_objs = len(gligen_boxes)

        tokenizer_inputs = self.tokenizer(
            gligen_phrases, padding=True, return_tensors="pt"
        )
        text_embeddings_raw = self.text_encoder(**tokenizer_inputs).pooler_output

        boxes = torch.zeros(max_objs, 4, dtype=dtype)
        boxes[:n_objs] = torch.tensor(gligen_boxes, dtype=dtype)

        text_embeddings = torch.zeros(max_objs, self.cross_attention_dim, dtype=dtype)
        text_embeddings[:n_objs] = text_embeddings_raw.to(dtype)

        masks = torch.zeros(max_objs, dtype=dtype)
        masks[:n_objs] = 1

        boxes = boxes.unsqueeze(0).expand(batch_size, -1, -1).clone()
        text_embeddings = (
            text_embeddings.unsqueeze(0).expand(batch_size, -1, -1).clone()
        )
        masks = masks.unsqueeze(0).expand(batch_size, -1).clone()

        cross_attention_kwargs = {
            "gligen": {
                "boxes": boxes,
                "positive_embeddings": text_embeddings,
                "masks": masks,
            }
        }

        return {
            "sample": latent_model_input,
            "timestep": self.scheduler.timesteps[0],
            "encoder_hidden_states": prompt_embeds,
            "cross_attention_kwargs": cross_attention_kwargs,
        }
