# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr model loader implementation for image-text-to-text (document OCR) tasks.

dots.ocr (rednote-hilab/dots.ocr) is a multimodal document-parsing model that
couples a NaViT-style ``dots_vit`` vision tower with a Qwen2-style causal-LM
decoder (``DotsOCRForCausalLM``). The model ships as custom code on the Hub, so
it is loaded with ``trust_remote_code=True`` pinned to a known revision.

This loader exercises the full vision + decoder forward pass. The vision tower
and the text decoder are also brought up independently via the sibling loaders
under ``dots_ocr/vision`` and ``dots_ocr/causal_lm``.
"""
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Optional

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

# Pinned Hub revision for the custom modeling code + weights (reproducibility).
DOTS_OCR_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"


def build_sample_image() -> Image.Image:
    """Build a small synthetic document image so the loader has no network
    dependency on an external image host."""
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "dots.ocr\nHello OCR\nInvoice #12345\nTotal: $42.00", fill="black")
    return img


class Wrapper(torch.nn.Module):
    """Returns logits as a single tensor for the test harness / PCC comparison."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        return outputs.logits


class ModelVariant(StrEnum):
    """Available dots.ocr variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """dots.ocr loader for the full image-text-to-text (document OCR) forward pass."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # OCR-style instruction paired with the synthetic document image.
    query = "Extract the text from this document."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=DOTS_OCR_REVISION,
        )
        return self.processor

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the full dots.ocr model wrapped to return logits.

        Args:
            dtype_override: Optional torch.dtype. dots.ocr's vision tower casts
                activations to bfloat16 internally, so bfloat16 is the natural
                dtype; defaults to bfloat16 when not provided.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model_kwargs = {
            "trust_remote_code": True,
            "revision": DOTS_OCR_REVISION,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()
        return Wrapper(model)

    def load_inputs(self, dtype_override=None):
        """Build sample image-text inputs for the full forward pass."""
        if self.processor is None:
            self._load_processor()

        image = build_sample_image()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.query},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        # Only the float pixel tensor takes the model dtype; ids/grids stay integer.
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }
