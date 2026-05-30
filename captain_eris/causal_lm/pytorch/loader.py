# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Captain-Eris-Diogenes_Twilight model loader implementation for causal language modeling.

This is a GGUF-quantized merge built on the Mistral-Nemo 12B architecture
(MistralForCausalLM). The weights are distributed only as GGUF files, so both the
model and tokenizer are loaded from a single GGUF file via the transformers
``gguf_file`` mechanism, which dequantizes the weights into a regular torch model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    """Available Captain-Eris-Diogenes_Twilight model variants."""

    Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Captain-Eris-Diogenes_Twilight loader for causal language modeling tasks."""

    # GGUF file (within the HF repo) backing each variant.
    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "Captain-Eris-Diogenes_Twilight-V0.420-12B-Q4_K_M.gguf",
    }

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/Captain-Eris-Diogenes_Twilight-V0.420-12B-GGUF",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the
                        model's default (full 40-layer model).
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.config = None
        self.num_layers = num_layers

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
            model="Captain-Eris-Diogenes_Twilight",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
            padding_side="right",
        )
        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the model instance for this instance's variant.

        The weights live only as GGUF, so we point ``from_pretrained`` at the
        repo plus the specific ``gguf_file``; transformers dequantizes them into
        a standard MistralForCausalLM. GGUF dequantization defaults to float32,
        which for a 12B model is large, so we default to bfloat16 unless the
        caller overrides the dtype.

        Args:
            dtype_override: Optional torch.dtype to override the default bfloat16.

        Returns:
            torch.nn.Module: The model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        model_kwargs["torch_dtype"] = (
            dtype_override if dtype_override is not None else torch.bfloat16
        )
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this variant's settings.

        Args:
            dtype_override: Optional torch.dtype (unused for integer token inputs;
                            accepted for interface compatibility).
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        test_input = "How often does the letter r occur in Mistral?"
        inputs = self.tokenizer(test_input, return_tensors="pt")

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        """Decode model outputs into human-readable next-token text.

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

    def load_config(self):
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
