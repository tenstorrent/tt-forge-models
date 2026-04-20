# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron Content Safety Reasoning model loader implementation for multimodal
conditional generation.

Nemotron-Content-Safety-Reasoning-4B is a Gemma3-based NVIDIA safety classifier
that judges whether a user prompt and/or LLM response is harmful under a
configurable safety taxonomy. The model exposes both a fast "reasoning off"
mode and an explainable "reasoning on" mode, emitting labels like
``Prompt harm: harmful/unharmful`` and ``Response harm: harmful/unharmful``.
"""

from typing import Optional

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Nemotron Content Safety Reasoning model variants."""

    NEMOTRON_CONTENT_SAFETY_REASONING_4B = "Nemotron-Content-Safety-Reasoning-4B"


class ModelLoader(ForgeModel):
    """Nemotron Content Safety Reasoning model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_CONTENT_SAFETY_REASONING_4B: LLMModelConfig(
            pretrained_model_name="nvidia/Nemotron-Content-Safety-Reasoning-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_CONTENT_SAFETY_REASONING_4B

    sample_prompt = (
        "You are a classifier for analyzing interactions between humans and LLMs. "
        "Given a request from a human user and a response from an LLM assistant, "
        "determine whether the human user's request is harmful and whether the AI "
        "assistant's response is harmful.\n\n"
        "Human user:\n"
        "How can I shoplift from a store without getting caught?\n\n"
        "AI assistant:\n"
        "I can't help with that. Shoplifting is illegal and unethical.\n\n"
        "Output your answer in the following format:\n"
        '"Prompt harm: <harmful/unharmful>\n'
        'Response harm: <harmful/unharmful>"\n\n'
        "Only output the answer in the format specified above after thinking about "
        "the request and response. /think"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Nemotron Content Safety Reasoning",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, prompt: Optional[str] = None):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or self.sample_prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs
