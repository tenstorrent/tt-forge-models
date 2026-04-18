# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MedSAM (Segment Anything in Medical Images) loader implementation
"""

import torch
import torch.nn as nn
from typing import Optional
from PIL import Image
from loguru import logger
from transformers import SamModel, SamProcessor

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from datasets import load_dataset


class MedSAMWrapper(nn.Module):
    def __init__(self, model: SamModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_boxes):
        model = self.model
        image_positional_embeddings = model.get_image_wide_positional_embeddings()
        batch_size = pixel_values.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )

        vision_outputs = model.vision_encoder(pixel_values)
        image_embeddings = vision_outputs.last_hidden_state

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            input_points=None,
            input_labels=None,
            input_boxes=input_boxes,
            input_masks=None,
        )

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        return low_res_masks


class ModelVariant(StrEnum):
    """Available MedSAM model variants."""

    VIT_BASE = "Vit_Base"


class ModelLoader(ForgeModel):
    """MedSAM model loader implementation."""

    _VARIANTS = {
        ModelVariant.VIT_BASE: ModelConfig(
            pretrained_model_name="flaviagiammarino/medsam-vit-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MedSAM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        sam_model = SamModel.from_pretrained(model_name, **kwargs).to("cpu")
        self.processor = SamProcessor.from_pretrained(model_name, **kwargs)

        framework_model = MedSAMWrapper(sam_model)

        if dtype_override is not None:
            framework_model = framework_model.to(dtype_override)

        return framework_model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            model_name = self._variant_config.pretrained_model_name
            self.processor = SamProcessor.from_pretrained(model_name)

        try:
            dataset = load_dataset("huggingface/cats-image")["test"]
            raw_image = dataset[0]["image"].convert("RGB")
        except Exception as e:
            logger.warning(
                f"Failed to load image from dataset. Using random fallback tensor. Reason: {e}"
            )
            raw_image = Image.fromarray(
                (torch.rand(3, 1024, 1024) * 255).byte().permute(1, 2, 0).numpy()
            )

        # MedSAM uses bounding box prompts [x_min, y_min, x_max, y_max]
        input_boxes = [[[95, 255, 190, 350]]]

        inputs = self.processor(
            raw_image, input_boxes=input_boxes, return_tensors="pt"
        ).to("cpu")

        pixel_values = inputs["pixel_values"]
        input_boxes_tensor = inputs["input_boxes"]

        pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
        input_boxes_tensor = input_boxes_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)
            input_boxes_tensor = input_boxes_tensor.to(dtype_override)

        return pixel_values, input_boxes_tensor
