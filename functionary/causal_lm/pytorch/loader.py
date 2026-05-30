# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Functionary model loader implementation for causal language modeling.

Functionary (meetkai/functionary-small-v2.4) is a Mistral-7B based chat model
tuned for function calling. The weights are distributed only in GGUF format, so
the model is loaded via transformers' GGUF dequantization path (gguf_file=...),
which produces a standard LlamaForCausalLM module with float32 weights.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Functionary model variants for causal LM."""

    SMALL_V2_4 = "small_v2.4"


class ModelLoader(ForgeModel):
    """Functionary model loader implementation for causal language modeling tasks."""

    # GGUF file within the repo to dequantize. The quantization level does not
    # affect output-comparison correctness (the golden and device runs share the
    # same dequantized weights); Q4_0 is the smallest file and fastest to fetch.
    _GGUF_FILES = {
        ModelVariant.SMALL_V2_4: "functionary-small-v2.4.Q4_0.gguf",
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.SMALL_V2_4: LLMModelConfig(
            pretrained_model_name="meetkai/functionary-small-v2.4-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SMALL_V2_4

    # Sample text for causal LM
    sample_text = "How often does the letter r occur in Mistral?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Functionary",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        # The GGUF repo also ships the HF tokenizer files, so AutoTokenizer can
        # load directly from the repo without the gguf_file argument.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Functionary model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to cast the model to. GGUF weights
                            are dequantized to float32; if provided, the model is
                            cast to this dtype after loading.

        Returns:
            torch.nn.Module: The Functionary model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            gguf_file=gguf_file,
            **kwargs,
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Functionary model.

        Args:
            dtype_override: Optional torch.dtype to cast input tensors to.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Mistral-based models use sliding window attention; match the window to
        # the input length so cache updates stay in bounds for short sequences.
        if (
            hasattr(self.model.config, "sliding_window")
            and self.model.config.sliding_window is not None
        ):
            self.model.config.sliding_window = inputs["input_ids"].shape[1]

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def decode_output(self, outputs):
        """Decode model outputs into the predicted next-token text.

        Args:
            outputs: Model output from a forward pass.

        Returns:
            str: Decoded next token text.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
