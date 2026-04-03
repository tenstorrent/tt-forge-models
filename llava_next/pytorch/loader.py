# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-NeXT (LLaVA 1.6) model loader for multimodal conditional generation.

Reference: https://github.com/LLaVA-VL/LLaVA-NeXT
LLaVA-NeXT (also referred to as LLaVA 1.6) improves upon LLaVA 1.5 with:
  - Higher image resolution (up to 4× tiles)
  - Better visual reasoning and OCR
  - Multiple LLM backbones: Mistral-7B, Vicuna-7B/13B

Bringup strategy for TT single-chip:
  - Full 7B model: kept as NOT_SUPPORTED_SKIP (full LLM too large / dynamic shapes)
  - VisionEncoder variant: wraps vision_tower + multi_modal_projector only.
    Takes pixel_values (B, N_tiles, 3, 336, 336) with fixed N_tiles=5 and
    produces projected image features (B, N_tiles*576, lm_hidden).
    All shapes are static — no dynamic token sequence lengths.
"""

from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

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
from ...tools.utils import cast_input_to_type


class LLaVANextVisionEncoder(nn.Module):
    """Vision encoder sub-model of LLaVA-NeXT.

    Wraps vision_tower (CLIP ViT-L/14@336) + multi_modal_projector (2-layer MLP)
    with a static-shape forward pass.

    The full model's get_image_features() uses dynamic image_num_patches logic.
    This wrapper bypasses that by operating on pre-tiled pixel_values directly,
    reshaping (B, N_tiles, C, H, W) → (B*N_tiles, C, H, W) before the vision
    tower and restoring after projection.

    Input:
        pixel_values: (B, N_tiles, C, H, W) — pre-tiled image patches at 336×336
    Output:
        image_features: (B, N_tiles * num_patches, lm_hidden_size) — projected
            vision features ready to be merged into the LLM token sequence
    """

    def __init__(self, full_model: LlavaNextForConditionalGeneration):
        super().__init__()
        self.vision_tower = full_model.model.vision_tower
        self.projector = full_model.model.multi_modal_projector
        vision_feature_layer = full_model.config.vision_feature_layer
        # Normalize to int (config stores -2 for second-to-last layer)
        if isinstance(vision_feature_layer, int):
            self._feature_layer = vision_feature_layer
        else:
            self._feature_layer = vision_feature_layer[-1]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, N_tiles, C, H, W) float tensor

        Returns:
            image_features: (B, N_tiles * num_patches, lm_hidden_size)
        """
        B, N, C, H, W = pixel_values.shape
        # Flatten tile dimension into batch for vision tower
        pv = pixel_values.view(B * N, C, H, W)
        # CLIP vision tower: returns hidden_states of shape (B*N, seq_len, hidden)
        vf = self.vision_tower(pv, output_hidden_states=True)
        feats = vf.hidden_states[self._feature_layer]  # (B*N, 1+num_patches, hidden)
        feats = feats[:, 1:]                            # remove CLS: (B*N, num_patches, hidden)
        # MLP projector: (B*N, num_patches, hidden) → (B*N, num_patches, lm_hidden)
        projected = self.projector(feats)
        num_patches = feats.shape[1]
        # Restore: (B, N*num_patches, lm_hidden)
        return projected.view(B, N * num_patches, projected.shape[-1])


class ModelVariant(StrEnum):
    """Available LLaVA-NeXT model variants."""

    LLAVA_NEXT_VISION_ENCODER = "LLaVA_NeXT_Vision_Encoder"
    LLAVA_NEXT_MISTRAL_7B = "LLaVA_NeXT_Mistral_7B"
    LLAVA_NEXT_VICUNA_7B = "LLaVA_NeXT_Vicuna_7B"


class ModelLoader(ForgeModel):
    """LLaVA-NeXT model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_NEXT_VISION_ENCODER: ModelConfig(
            pretrained_model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        ),
        ModelVariant.LLAVA_NEXT_MISTRAL_7B: ModelConfig(
            pretrained_model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        ),
        ModelVariant.LLAVA_NEXT_VICUNA_7B: ModelConfig(
            pretrained_model_name="llava-hf/llava-v1.6-vicuna-7b-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_NEXT_VISION_ENCODER

    # Fixed tile count: cats-image at 480×640 always produces 5 tiles via anyres
    _NUM_TILES = 5
    # CLIP ViT-L/14@336: (336/14)^2 = 576 patches per tile
    _PATCHES_PER_TILE = 576

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-NeXT",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = LlavaNextProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the model instance.

        For LLAVA_NEXT_VISION_ENCODER: returns LLaVANextVisionEncoder (vision_tower +
            multi_modal_projector only, ~300M params, static shapes, single-chip compatible).
        For LLAVA_NEXT_MISTRAL_7B / LLAVA_NEXT_VICUNA_7B: returns the full
            LlavaNextForConditionalGeneration (7B, kept as NOT_SUPPORTED_SKIP in test config).

        Args:
            dtype_override: Optional torch.dtype override.

        Returns:
            torch.nn.Module: model instance.
        """
        model_name = self._variant_config.pretrained_model_name
        full_model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, low_cpu_mem_usage=True, **kwargs
        )
        full_model.eval()

        if dtype_override:
            full_model = full_model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        if self._variant == ModelVariant.LLAVA_NEXT_VISION_ENCODER:
            model = LLaVANextVisionEncoder(full_model)
            model.eval()
            return model

        return full_model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors.

        For LLAVA_NEXT_VISION_ENCODER: returns pixel_values (B, N_tiles, C, H, W)
            with N_tiles=5 (fixed from cats-image 480×640 via anyres tiling).
        For full model variants: returns dict with input_ids, attention_mask,
            pixel_values, image_sizes.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).

        Returns:
            For vision encoder: torch.Tensor (B, N_tiles, 3, 336, 336)
            For full model: dict with 'input_ids', 'attention_mask', 'pixel_values', 'image_sizes'
        """
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        )

        if self._variant == ModelVariant.LLAVA_NEXT_VISION_ENCODER:
            pixel_values = inputs["pixel_values"]  # (1, N_tiles, 3, 336, 336)
            if batch_size > 1:
                pixel_values = pixel_values.expand(batch_size, -1, -1, -1, -1)
            if dtype_override:
                pixel_values = pixel_values.to(dtype_override)
            return pixel_values

        # Full model inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs.get("image_sizes")

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        if image_sizes is not None:
            result["image_sizes"] = image_sizes

        return result
