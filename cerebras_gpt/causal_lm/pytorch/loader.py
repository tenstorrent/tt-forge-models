# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cerebras-GPT model loader implementation for causal language modeling.

Cerebras-GPT models use the GPT-2 architecture (model_type "gpt2"), so they load
through the standard transformers Auto classes.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
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
    """Available Cerebras-GPT model variants."""

    GPT_1_3B = "1_3b"


class ModelLoader(ForgeModel):
    """Cerebras-GPT loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GPT_1_3B: LLMModelConfig(
            pretrained_model_name="cerebras/Cerebras-GPT-1.3B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_1_3B

    sample_text = "Generative AI is"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Cerebras-GPT",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cerebras-GPT model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The loaded model instance.
        """
        model_name = self._variant_config.pretrained_model_name

        config = GPT2Config.from_pretrained(model_name)
        config_dict = config.to_dict()
        config_dict["use_cache"] = False
        if dtype_override is not None:
            config_dict["torch_dtype"] = dtype_override
        config = GPT2Config(**config_dict)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Cerebras-GPT model.

        Args:
            dtype_override: Optional torch.dtype (unused for integer input ids).

        Returns:
            dict: Tokenized inputs with input_ids and attention_mask.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
