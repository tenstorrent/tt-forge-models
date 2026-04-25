# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf model loader implementation
for multimodal conditional generation.
"""

from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import (
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
)


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to map llava HF model type to llama GGUF arch.

    LLaVA GGUF files use the llama tensor layout for the language backbone, but
    LlavaConfig reports model_type='llava' which gguf-py does not recognise.
    """
    effective_type = hf_model.config.model_type if model_type is None else model_type
    if effective_type == "llava":
        model_type = "llama"
        if num_layers is None:
            num_layers = hf_model.config.text_config.num_hidden_layers
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoConfig

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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available llama-joycaption-beta-one-hf-llava-mmproj-gguf model variants."""

    BETA_ONE_HF_LLAVA_Q4_K = "Beta_One_HF_LLaVA_Q4_K"


class ModelLoader(ForgeModel):
    """llama-joycaption-beta-one-hf-llava-mmproj-gguf model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.BETA_ONE_HF_LLAVA_Q4_K: ModelConfig(
            pretrained_model_name="concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BETA_ONE_HF_LLAVA_Q4_K

    _GGUF_FILES = {
        ModelVariant.BETA_ONE_HF_LLAVA_Q4_K: "Llama-Joycaption-Beta-One-Hf-Llava-Q4_K.gguf",
    }

    _PROCESSOR_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"

    sample_text = "Write a long descriptive caption for this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize llama-joycaption-beta-one-hf-llava-mmproj-gguf model loader."""
        super().__init__(variant)
        self.processor = None
        self.config = None

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="llama-joycaption-beta-one-hf-llava-mmproj-gguf",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._PROCESSOR_NAME, use_fast=False
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the llama-joycaption-beta-one-hf-llava-mmproj-gguf model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        # The concedo repo has an outdated config; use the original full model config
        # so the architecture matches the Llama 3.1-based GGUF weights.
        config = AutoConfig.from_pretrained(self._PROCESSOR_NAME)
        model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
            **model_kwargs,
        ).eval()

        self.config = model.config

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for llama-joycaption-beta-one-hf-llava-mmproj-gguf."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": f"<image>\n{self.sample_text}",
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
