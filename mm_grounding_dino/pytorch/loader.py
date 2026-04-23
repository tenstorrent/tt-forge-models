# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MM Grounding DINO model loader implementation for zero-shot object detection.
"""
import math

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from typing import Optional

# Patch get_sine_pos_embed to use pos_tensor.dtype instead of hardcoded float32.
# Without this patch, bfloat16 hidden_states + float32 position_embeddings produces
# float32 queries which then fail against bfloat16 linear layer weights.
import transformers.models.mm_grounding_dino.modeling_mm_grounding_dino as _mm_dino_mod


def _patched_get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
) -> torch.Tensor:
    scale = 2 * math.pi
    dim_t = torch.arange(
        num_pos_feats, dtype=pos_tensor.dtype, device=pos_tensor.device
    )
    dim_t = temperature ** (
        2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
    )

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack(
            (sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3
        ).flatten(2)
        return sin_x

    pos_tensor = pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    position_embeddings = [sine_func(x) for x in pos_tensor]
    if exchange_xy:
        position_embeddings[0], position_embeddings[1] = (
            position_embeddings[1],
            position_embeddings[0],
        )
    position_embeddings = torch.cat(position_embeddings, dim=-1)
    return position_embeddings


_mm_dino_mod.get_sine_pos_embed = _patched_get_sine_pos_embed

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


class ModelVariant(StrEnum):
    """Available MM Grounding DINO model variants for zero-shot object detection."""

    BASE_ALL = "Base_All"
    LARGE_ALL = "Large_All"
    TINY_O365V1_GOLDG_V3DET = "Tiny_O365V1_GoldG_V3Det"


class ModelLoader(ForgeModel):
    """MM Grounding DINO model loader implementation for zero-shot object detection tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_ALL: ModelConfig(
            pretrained_model_name="openmmlab-community/mm_grounding_dino_base_all",
        ),
        ModelVariant.LARGE_ALL: ModelConfig(
            pretrained_model_name="openmmlab-community/mm_grounding_dino_large_all",
        ),
        ModelVariant.TINY_O365V1_GOLDG_V3DET: ModelConfig(
            pretrained_model_name="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE_ALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.image = None
        self.text_labels = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="MM-Grounding-DINO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_ZS_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MM Grounding DINO model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MM Grounding DINO model instance for zero-shot object detection.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the MM Grounding DINO model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        self.image = Image.new("RGB", (640, 480))

        self.text_labels = [["a cat", "a remote control"]]

        inputs = self.processor(
            images=self.image, text=self.text_labels, return_tensors="pt"
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
