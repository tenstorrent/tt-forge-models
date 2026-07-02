# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr model loader implementation for multimodal document-OCR tasks.

dots.ocr (rednote-hilab/dots.ocr) is a compact vision-language OCR model: a
NaViT-style vision transformer (``DotsVisionTransformer``) feeds a Qwen2
decoder-only language model (``DotsOCRForCausalLM`` subclasses
``Qwen2ForCausalLM``). It is distributed as a ``trust_remote_code`` model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

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
from .src.model import DotsOCRWrapper


class ModelVariant(StrEnum):
    """Available dots.ocr model variants."""

    DOTS_OCR = "dots_ocr"


class ModelLoader(ForgeModel):
    """dots.ocr model loader implementation for multimodal document-OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DOTS_OCR: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DOTS_OCR

    # Pin the trust_remote_code checkpoint so the custom modeling files
    # (modeling_dots_ocr.py / modeling_dots_vision.py) don't drift between runs.
    REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"

    # Official dots.ocr layout+OCR prompt (from the model card). Drives the
    # model to extract layout elements and their text from the document image.
    sample_prompt = (
        "Please output the layout information from the PDF image, including each "
        "layout element's bbox, its category, and the corresponding text content "
        "within the bbox.\n\n"
        "1. Bbox format: [x1, y1, x2, y2]\n\n"
        "2. Layout Categories: The possible categories are ['Caption', 'Footnote', "
        "'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', "
        "'Section-header', 'Table', 'Text', 'Title'].\n\n"
        "3. Text Extraction & Formatting Rules:\n"
        "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
        "    - Formula: Format its text as LaTeX.\n"
        "    - Table: Format its text as HTML.\n"
        "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
        "4. Constraints:\n"
        "    - The output text must be the original text from the image, with no "
        "translation.\n"
        "    - All layout elements must be sorted according to human reading order.\n\n"
        "5. Final Output: The entire output must be a single JSON object.\n"
    )

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
        """Load the DotsVLProcessor for the current variant.

        Returns:
            The loaded processor instance.
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=self.REVISION,
        )
        return self.processor

    @staticmethod
    def _build_document_image() -> Image.Image:
        """Build a small, self-contained synthetic document image.

        Kept dependency-free (no network / CI cache) so the loader is
        reproducible; the drawn text is large enough to exercise the OCR path.
        """
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        lines = [
            "INVOICE  No. 12345",
            "Date: 2026-07-02",
            "Bill To: Tenstorrent AI",
            "",
            "Item          Qty     Price",
            "Widget A       2      $10.00",
            "Widget B       1      $25.00",
            "",
            "Total:                $45.00",
            "Thank you for your business!",
        ]
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22
            )
        except Exception:
            font = ImageFont.load_default()
        y = 30
        for line in lines:
            draw.text((40, y), line, fill="black", font=font)
            y += 40
        return image

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the dots.ocr model instance for this instance's variant.

        The model is loaded in bfloat16 by default: the vision tower's forward
        internally casts activations to bfloat16 (``bf16=True``), so a float32
        run would mismatch the bfloat16 patch-embed weights. bfloat16 also
        matches the checkpoint's native ``torch_dtype`` and the device run.

        Args:
            dtype_override: Optional torch.dtype to override the model dtype.

        Returns:
            torch.nn.Module: The wrapped dots.ocr model returning logits.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            revision=self.REVISION,
            torch_dtype=dtype,
            **kwargs,
        )
        self.config = model.config
        model.config.use_cache = False
        model.eval()

        return DotsOCRWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the dots.ocr model.

        Args:
            dtype_override: Optional torch.dtype for the pixel_values tensor.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors (input_ids, attention_mask, pixel_values,
                  image_grid_thw) that can be fed to the wrapped model.
        """
        if self.processor is None:
            self._load_processor()

        image = self._build_document_image()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text] * batch_size,
            images=[image] * batch_size,
            padding=True,
            return_tensors="pt",
        )

        # The model forward does not consume mm_token_type_ids emitted by the
        # processor; drop it so only the accepted keys are passed.
        inputs.pop("mm_token_type_ids", None)

        # Cast pixel_values to the model dtype (default bfloat16).
        pixel_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        inputs["pixel_values"] = inputs["pixel_values"].to(pixel_dtype)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }

    def load_config(self):
        """Load and return the HF configuration for the dots.ocr variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=self.REVISION,
        )
        return self.config
