# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 model loader implementation for multimodal (image-text-to-text)
conditional generation.

Molmo2-8B is a vision-language model from AllenAI: a SigLIP-style vision
transformer + an image-pooling/projector adapter feeding a Qwen3-8B-derived
decoder-only language model. The checkpoint ships custom modeling code
(``trust_remote_code=True``) authored against transformers 4.57.x, so the
revision and a matching ``transformers`` pin are required (see requirements.txt).
"""

from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

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

# Pinned commit of the custom modeling/processing code on the Hub. The model
# code targets transformers==4.57.1; newer transformers drop APIs it relies on
# (ROPE_INIT_FUNCTIONS["default"], the processor's optional attributes).
_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"


class ModelVariant(StrEnum):
    """Available Molmo2 model variants."""

    MOLMO2_8B = "8B"


class ModelLoader(ForgeModel):
    """Molmo2 loader for multimodal conditional generation (image+text -> text)."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: ModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Molmo2 model loader."""
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=_REVISION,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Molmo2ForConditionalGeneration model instance."""
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=_REVISION,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        model.eval()
        # Disable KV cache so the traced graph has no Cache object in its output.
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False

        self.config = model.config
        if self.processor is None:
            self._load_processor()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build multimodal (image + text) inputs via the Molmo2 processor."""
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.sample_text},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # token_type_ids is not consumed by the forward pass; drop it to keep the
        # traced graph clean.
        inputs.pop("token_type_ids", None)

        result = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": cast_input_to_type(
                inputs["pixel_values"], dtype_override
            ),
            "image_token_pooling": inputs["image_token_pooling"],
            "image_grids": inputs["image_grids"],
            "image_num_crops": inputs["image_num_crops"],
        }
        return result

    def load_config(self):
        """Load and return the configuration for the model variant."""
        if self.config is None:
            self.load_model()
        return self.config
