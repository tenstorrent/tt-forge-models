# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
German GPT-2 model loader implementation for causal language modeling.
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available German GPT-2 model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """German GPT-2 loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="anonymous-german-nlp/german-gpt2",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # German sample text for causal LM
    sample_text = "Heute ist ein schöner Tag und die Sonne"

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
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="German-GPT2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load the tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the German GPT-2 model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The German GPT-2 model instance for causal LM.
        """
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the German GPT-2 model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' dtype.
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) suitable for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # NOTE: This model's tokenizer defines an eos token (id 52000) that lies
        # outside the model's embedding table (vocab_size=52000, valid ids 0..51999).
        # Padding with eos would index out of range, so we tokenize without padding
        # and rely on the natural sequence length.
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        # Replicate tensors for the requested batch size.
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested (applies to floating tensors).
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return dict(inputs)

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Model outputs (with a ``logits`` attribute or tuple).
            inputs: Optional inputs used to produce the outputs.

        Returns:
            str: The decoded text from the model's argmax predictions.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
