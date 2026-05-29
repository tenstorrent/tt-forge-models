# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
A0l-12B-heretic model loader implementation for causal language modeling.

The model is distributed only as GGUF-quantized files
(``mradermacher/A0l-12B-heretic-GGUF``). We load it through HuggingFace
transformers' GGUF support, which dequantizes the chosen GGUF file back into a
standard PyTorch ``MistralForCausalLM`` (the underlying base model
``nbeerbower/A0l-12B-heretic`` is a Mistral-Nemo 12B architecture).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available A0l-12B-heretic GGUF variants."""

    Q4_K_M = "q4_k_m"


class ModelLoader(ForgeModel):
    """A0l-12B-heretic GGUF loader for causal language modeling tasks."""

    # GGUF filename within the HF repo for each variant.
    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "A0l-12B-heretic.Q4_K_M.gguf",
    }

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/A0l-12B-heretic-GGUF",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

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
            model="A0l-12B-heretic",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load the tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the A0l-12B-heretic model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.
                            If not provided, the model uses bfloat16.

        Returns:
            torch.nn.Module: The model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer()

        # Default to bfloat16 for hardware; the GGUF path otherwise dequantizes
        # the weights to float32.
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            gguf_file=gguf_file,
            torch_dtype=dtype,
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this variant.

        Args:
            dtype_override: Optional torch.dtype (unused for integer token ids).
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Keep the sequence short to fit the 12B model on a single device.
        test_input = "The future of artificial intelligence is"
        inputs = self.tokenizer(test_input, return_tensors="pt")

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded next token text
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
