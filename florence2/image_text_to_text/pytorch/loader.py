# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Florence-2 model loader implementation for image-text-to-text generation.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available Florence-2 model variants."""

    FLORENCE_2_LARGE_FT = "microsoft/Florence-2-large-ft"


class ModelLoader(ForgeModel):
    """Florence-2 model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.FLORENCE_2_LARGE_FT: ModelConfig(
            pretrained_model_name=str(ModelVariant.FLORENCE_2_LARGE_FT),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLORENCE_2_LARGE_FT

    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    sample_text = "<OD>"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Florence-2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        pretrained_name = self._variant_config.pretrained_model_name
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name, trust_remote_code=True
        )
        if not hasattr(tokenizer, "additional_special_tokens"):
            tokenizer.additional_special_tokens = list(
                getattr(tokenizer, "_extra_special_tokens", [])
            )
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_name, trust_remote_code=True, use_fast=False
        )
        processor_class = get_class_from_dynamic_module(
            "processing_florence2.Florence2Processor",
            pretrained_name,
            trust_remote_code=True,
        )
        self.processor = processor_class(
            image_processor=image_processor, tokenizer=tokenizer
        )
        return self.processor

    @staticmethod
    def _patch_florence2_config(pretrained_name):
        lang_cfg = get_class_from_dynamic_module(
            "configuration_florence2.Florence2LanguageConfig",
            pretrained_name,
            trust_remote_code=True,
        )
        if not hasattr(lang_cfg, "forced_bos_token_id"):
            lang_cfg.forced_bos_token_id = None

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Florence-2 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        self._patch_florence2_config(pretrained_model_name)

        model_kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Florence-2."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        inputs = self.processor(
            text=self.sample_text,
            images=image,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        decoder_start_id = self.processor.tokenizer.convert_tokens_to_ids("</s>")
        inputs["decoder_input_ids"] = torch.tensor(
            [[decoder_start_id]], dtype=torch.long
        )

        return inputs
