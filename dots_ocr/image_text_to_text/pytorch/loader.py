# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr loader - full multimodal document-OCR pipeline (image-text-to-text).

rednote-hilab/dots.ocr is a ``DotsOCRForCausalLM`` model: a NaViT-style
``dots_vit`` vision tower (42 layers) whose patch embeddings are spliced into a
Qwen2 text decoder (28 layers) via the image token. The modeling code is loaded
through ``trust_remote_code``; the revision is pinned in ``_common`` so it stays
reproducible.
"""
import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoProcessor

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
from ..._common import DOTS_OCR_MODEL, DOTS_OCR_REVISION, build_demo_image
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available dots.ocr variants."""

    BASE = "1.7b"


class ModelLoader(ForgeModel):
    """Loader for the full dots.ocr image-text-to-text pipeline."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name=DOTS_OCR_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # OCR-style instruction. dots.ocr is trained to transcribe document text.
    prompt = "Extract the text from this image."

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
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, dtype_override=None):
        """Load the full dots.ocr model wrapped to return logits.

        The vision tower defaults to flash_attention_2 (unavailable here) and
        falls back to an eager O(n^2) attention; we set it explicitly to eager
        so the device path is a plain matmul/softmax. The text decoder is set to
        eager attention as well.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        # Vision tower attention implementation is read per-block from the
        # vision config; force eager (plain matmul/softmax) for the device path.
        model.config.vision_config.attn_implementation = "eager"
        model.config.use_cache = False
        model.eval()

        return Wrapper(model)

    def load_inputs(self, dtype_override=None):
        """Build image+text inputs for the OCR pipeline.

        Uses a small (392x392) synthetic document image so the vision sequence
        stays short. Returns the four tensors the wrapped forward consumes.
        """
        if self.processor is None:
            self._load_processor()

        image = build_demo_image()
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
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)
        else:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }
