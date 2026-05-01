# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MedGemma GGUF model loader implementation for multimodal conditional generation.
"""
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoConfig,
    Gemma3Config,
    Gemma3ForConditionalGeneration,
)
from typing import Optional

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
from PIL import Image


class ModelVariant(StrEnum):
    """Available MedGemma GGUF model variants."""

    MEDGEMMA_1_5_4B_IT_Q4_K_M = "1.5_4B_IT_Q4_K_M"


class ModelLoader(ForgeModel):
    """MedGemma GGUF model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.MEDGEMMA_1_5_4B_IT_Q4_K_M: ModelConfig(
            pretrained_model_name="unsloth/medgemma-1.5-4b-it-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEDGEMMA_1_5_4B_IT_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.MEDGEMMA_1_5_4B_IT_Q4_K_M: "medgemma-1.5-4b-it-Q4_K_M.gguf",
    }

    # Public (non-gated) source for processor and full multimodal config.
    # The GGUF only contains text backbone; the multimodal config with the
    # SigLIP vision config comes from this repo.
    _PROCESSOR_NAME = "unsloth/medgemma-1.5-4b-it"

    sample_text = "Describe any abnormalities in this medical image."
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize MedGemma GGUF model loader."""
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
            model="MedGemma GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override
        # transformers 5.x defaults to use_fast=True for Gemma3ImageProcessor;
        # use_fast=False preserves the slow (compatible) processor behavior.
        self.processor = AutoProcessor.from_pretrained(
            self._PROCESSOR_NAME, use_fast=False, **kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MedGemma GGUF model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # The medgemma GGUF only contains the text backbone (Gemma3TextConfig).
        # transformers 5.x explicitly remaps gemma3 GGUF → gemma3_text, so
        # AutoModelForImageTextToText refuses to load it.  Work around by:
        #  1. Loading Gemma3ForCausalLM from the GGUF (text weights only).
        #  2. Creating Gemma3ForConditionalGeneration from the full multimodal
        #     config (includes SiglipVisionConfig from _PROCESSOR_NAME).
        #  3. Copying text weights into language_model.* of the full model.
        #     The vision_tower and multi_modal_projector are left at their
        #     random init values, which is acceptable for compiler testing
        #     (CPU and TT run the same random weights so PCC still measures
        #     compiler correctness, not vision accuracy).
        text_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            gguf_file=self.gguf_file,
            torch_dtype=dtype,
        )

        full_config = Gemma3Config.from_pretrained(self._PROCESSOR_NAME)
        model = Gemma3ForConditionalGeneration(full_config).to(dtype)

        # Map Gemma3ForCausalLM keys → Gemma3ForConditionalGeneration keys:
        #   model.X  →  model.language_model.X
        #   lm_head.X  →  lm_head.X  (same)
        full_sd = model.state_dict()
        for name, param in text_model.state_dict().items():
            if name.startswith("model."):
                target = "model.language_model." + name[len("model."):]
            else:
                target = name
            if target in full_sd:
                full_sd[target].copy_(param)
        model.load_state_dict(full_sd)
        del text_model

        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for MedGemma GGUF."""
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def load_config(self):
        """Load and return the model configuration."""
        self.config = Gemma3Config.from_pretrained(self._PROCESSOR_NAME)
        return self.config
