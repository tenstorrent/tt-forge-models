# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr vision-tower loader (image-feature bringup).

dots.ocr's vision tower is ``DotsVisionTransformer`` (``dots_vit``): a NaViT-style
ViT with a Conv2d patch embed (kernel=stride=14), 2D rotary position embeddings,
eager block attention (flash-attn falls back to eager when unavailable), and a
spatial 2x2 patch merger. This loader brings up the vision tower in isolation so
the (convolution-heavy) image encoder can be validated independently of the
text decoder, mirroring the per-component bringup of a multimodal pipeline.
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
    """Synthetic document image so the loader has no external-host dependency."""
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "dots.ocr\nHello OCR\nInvoice #12345\nTotal: $42.00", fill="black")
    return img


class VisionWrapper(torch.nn.Module):
    """Runs only the vision tower and returns image embeddings (single tensor)."""

    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower

    def forward(self, pixel_values, image_grid_thw):
        return self.vision_tower(pixel_values, image_grid_thw)


class ModelVariant(StrEnum):
    """Available dots.ocr vision-tower variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """dots.ocr vision-tower (dots_vit) loader."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

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
            task=ModelTask.CV_IMAGE_FE,
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
        """Load the dots.ocr model and return only its vision tower (wrapped)."""
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
        model.eval()
        return VisionWrapper(model.vision_tower)

    def load_inputs(self, dtype_override=None):
        """Build flattened patch pixel_values + grid_thw for the vision tower."""
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
        return {
            "pixel_values": inputs["pixel_values"].to(dtype),
            "image_grid_thw": inputs["image_grid_thw"],
        }
