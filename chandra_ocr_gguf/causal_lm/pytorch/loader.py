# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chandra OCR GGUF model loader implementation for causal language modeling.

Note: The qwen3vl GGUF architecture is not yet supported by the transformers
GGUF loader, so we load from the base safetensors checkpoint (datalab-to/chandra)
and extract the causal LM component.
"""
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)
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


class ModelVariant(StrEnum):
    """Available Chandra OCR GGUF model variants for causal language modeling."""

    CHANDRA_OCR_GGUF = "chandra_OCR_GGUF"


class ModelLoader(ForgeModel):
    """Chandra OCR GGUF model loader implementation for causal language modeling tasks.

    Note: Uses the base model (safetensors) instead of GGUF because the
    qwen3vl GGUF architecture is not yet supported by transformers.
    """

    _VARIANTS = {
        ModelVariant.CHANDRA_OCR_GGUF: LLMModelConfig(
            pretrained_model_name="datalab-to/chandra",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHANDRA_OCR_GGUF

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Chandra OCR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load the full conditional generation model directly; pixel_values are
        # optional so the model can be called with text-only inputs.
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        if self.num_layers is not None:
            model.config.text_config.num_hidden_layers = self.num_layers
        model.eval()

        self.config = model.config.text_config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
        self.config = config.text_config if hasattr(config, "text_config") else config
        return self.config
