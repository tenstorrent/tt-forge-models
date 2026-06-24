# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr model loader implementation for vision-language OCR / document-parsing tasks.

dots.ocr (rednote-hilab/dots.ocr) couples a Qwen2-style decoder-only language
model (1.5B) with a 42-layer NaViT-style vision transformer (``dots_vit``). It is
distributed as a HuggingFace ``custom_code`` checkpoint, so the model and
processor are loaded with ``trust_remote_code=True``.
"""
import types
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class _LinearPatchEmbed(nn.Module):
    """Drop-in replacement for dots.ocr's Conv2d patch embedding.

    The vision tower patchifies the image with a ``Conv2d`` whose stride equals
    its kernel size (14) and has no padding, so it only ever sees disjoint
    14x14 patches — i.e. it is exactly an ``nn.Linear`` over each flattened
    patch. The equivalent matmul avoids the ttnn conv2d path, which fails at
    runtime for this (3 -> 1536, kernel/stride 14) shape with
    "Reader indices buffer page size 132 exceeds worst-case CB size 64".
    Weights are copied from the original conv, so the math is unchanged.
    """

    def __init__(self, patch_embed):
        super().__init__()
        self.num_channels = patch_embed.num_channels
        self.patch_size = patch_embed.patch_size
        self.temporal_patch_size = patch_embed.temporal_patch_size
        self.embed_dim = patch_embed.embed_dim

        conv = patch_embed.proj  # nn.Conv2d(3, embed_dim, kernel=stride=patch_size)
        in_features = self.num_channels * self.patch_size * self.patch_size
        linear = nn.Linear(
            in_features, self.embed_dim, bias=conv.bias is not None
        )
        with torch.no_grad():
            # Conv weight [embed_dim, C, kH, kW] flattens (C, kH, kW) in the same
            # C-contiguous order as the input patch reshape below.
            linear.weight.copy_(conv.weight.reshape(self.embed_dim, in_features))
            if conv.bias is not None:
                linear.bias.copy_(conv.bias)
        self.proj = linear.to(conv.weight.dtype)
        self.norm = patch_embed.norm

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        x = x.view(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )[:, :, 0]
        x = x.reshape(-1, self.num_channels * self.patch_size * self.patch_size)
        x = self.proj(x).view(-1, self.embed_dim)
        x = self.norm(x)
        return x


def _replace_patch_conv_with_linear(model):
    """Swap the vision tower's Conv2d patchifier for the matmul equivalent."""
    preprocessor = model.vision_tower.patch_embed
    preprocessor.patchifier = _LinearPatchEmbed(preprocessor.patchifier)
    return model


def _disable_vision_bf16_cast(model):
    """Stop the vision tower from force-casting activations to bfloat16.

    ``DotsVisionTransformer.forward`` casts its input to bf16 when ``bf16=True``
    (the default), which both clashes with a float32 model and pins the entire
    vision trunk to bf16 precision. Forcing ``bf16=False`` lets the vision tower
    run in the model's own dtype.
    """
    vision_tower = model.vision_tower
    base_forward = type(vision_tower).forward

    def forward(self, hidden_states, grid_thw, bf16=False):
        return base_forward(self, hidden_states, grid_thw, bf16=False)

    vision_tower.forward = types.MethodType(forward, vision_tower)
    return model


class ModelVariant(StrEnum):
    """Available dots.ocr model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """dots.ocr model loader implementation for vision-language OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Prompt fed alongside the document image. dots.ocr is an OCR / layout model;
    # for bringup we only need a valid forward pass, so a simple instruction is fine.
    prompt = "Extract the text from this image."

    # Cap the number of vision patches so the (image + text) sequence stays small
    # enough to compile comfortably on a single chip. The Qwen2-VL-style image
    # processor resizes to a multiple of patch_size * spatial_merge_size (28).
    min_pixels = 28 * 28 * 4
    max_pixels = 28 * 28 * 256

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

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
            model="dots.ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load the processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
        )
        return self.processor

    def _build_sample_image(self):
        """Build a small synthetic document image for a deterministic forward pass.

        Avoids depending on network access to an external image at test time.
        """
        image = Image.new("RGB", (224, 224), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 90), "Tenstorrent dots.ocr", fill="black")
        draw.text((20, 120), "Hello, world!", fill="black")
        return image

    def load_model(self, dtype_override=None):
        """Load and return the dots.ocr model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The wrapped dots.ocr model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False

        # Rewrite the vision patch-embedding Conv2d as an equivalent matmul to
        # avoid the unsupported ttnn conv2d path for this patchify shape.
        model = _replace_patch_conv_with_linear(model)
        # Keep the vision trunk in the model's dtype instead of forcing bf16.
        model = _disable_vision_bf16_cast(model)

        model.eval()

        model = Wrapper(model)
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the dots.ocr model.

        Args:
            dtype_override: Optional torch.dtype to override the floating-point
                            inputs' default dtype (applied to pixel_values).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        image = self._build_sample_image()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Only the floating-point pixel_values follow the model dtype; the
        # integer ids / grid tensors must stay integral.
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # Keep only the tensors DotsOCRForCausalLM.forward consumes. The
        # processor may emit extras (e.g. mm_token_type_ids) that the model
        # does not accept; the image/text fusion is driven by image_token_id.
        keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        return {k: inputs[k] for k in keep}
