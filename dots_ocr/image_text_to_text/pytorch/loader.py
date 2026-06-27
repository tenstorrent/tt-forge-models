# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr full multimodal loader for document OCR (image-text-to-text).

dots.ocr (``DotsOCRForCausalLM``) couples a ``DotsVisionTransformer`` patch-embed
image encoder with a ``Qwen2ForCausalLM`` decoder. This loader exercises the full
multimodal forward: image patches flow through the vision tower, are merged into
the text token stream at the image-placeholder positions, and the decoder produces
text logits. The text-decoder-only path lives in
``dots_ocr/causal_lm/pytorch/loader.py``.
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from typing import Optional
from PIL import Image, ImageDraw

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

# Pin the snapshot so the trust_remote_code modeling/config files are reproducible.
DOTS_OCR_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"

# The processor (Qwen2.5-VL-based DotsVLProcessor) emits an extra key the forward
# does not consume; keep only the tensors DotsOCRForCausalLM.forward accepts.
_FORWARD_INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")


class ModelVariant(StrEnum):
    """Available dots.ocr variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """dots.ocr full multimodal loader for document OCR (image-text-to-text)."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    sample_prompt = "Extract the text from the image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None

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
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
        )
        return self.processor

    @staticmethod
    def _sample_image():
        """Build a deterministic synthetic document image (no network dependency).

        A white page with a few lines of black text is enough to exercise the
        patch-embed conv and the full vision tower.
        """
        img = Image.new("RGB", (448, 448), color="white")
        draw = ImageDraw.Draw(img)
        lines = [
            "Tenstorrent dots.ocr bringup",
            "The quick brown fox jumps",
            "over the lazy dog. 0123456789",
        ]
        for i, line in enumerate(lines):
            draw.text((20, 40 + i * 60), line, fill="black")
        return img

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the dots.ocr model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The DotsOCRForCausalLM instance (full multimodal path).
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True, "revision": DOTS_OCR_REVISION}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return full multimodal sample inputs (image + text).

        Args:
            dtype_override: Optional torch.dtype to override pixel_values dtype.

        Returns:
            dict: Input tensors (input_ids, attention_mask, pixel_values,
                  image_grid_thw) that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        image = self._sample_image()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_prompt},
                ],
            }
        ]

        # Chat template emits a single image-pad placeholder; processor(text, images)
        # expands it to the patch count implied by image_grid_thw.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # Drop processor extras (e.g. mm_token_type_ids) the forward does not accept.
        inputs = {k: v for k, v in inputs.items() if k in _FORWARD_INPUT_KEYS}

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def load_config(self):
        """Load and return the configuration for the dots.ocr model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
        )
        return self.config
