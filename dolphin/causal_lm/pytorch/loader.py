# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dolphin model loader implementation for causal language modeling.

Dolphin 2.8 Mistral 7B v0.2 is distributed only as GGUF (llama.cpp) quantized
weights. transformers can load a GGUF checkpoint directly via the ``gguf_file``
argument, dequantizing the weights back into a standard Mistral PyTorch model.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import torch

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
    """Available Dolphin model variants for causal LM."""

    # Dolphin 2.8 Mistral 7B v0.2, GGUF Q4_K_M quantization.
    DOLPHIN_2_8_MISTRAL_7B_V02_Q4_K_M = "2.8_mistral_7b_v02_q4_k_m"


class ModelLoader(ForgeModel):
    """Dolphin model loader implementation for causal language modeling tasks."""

    # GGUF filename (within the HF repo) backing each variant.
    _GGUF_FILES = {
        ModelVariant.DOLPHIN_2_8_MISTRAL_7B_V02_Q4_K_M: "dolphin-2.8-mistral-7b-v02-Q4_K_M.gguf",
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DOLPHIN_2_8_MISTRAL_7B_V02_Q4_K_M: LLMModelConfig(
            pretrained_model_name="lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DOLPHIN_2_8_MISTRAL_7B_V02_Q4_K_M

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

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
            model="Dolphin",
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
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # The repo ships no HF tokenizer files; the tokenizer is embedded in the
        # GGUF metadata and is recovered via the gguf_file argument.
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Dolphin model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model is loaded in bfloat16.

        Returns:
            torch.nn.Module: The Dolphin model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        # Default to bfloat16 for Tenstorrent hardware; the GGUF checkpoint is
        # otherwise dequantized to float32.
        model_kwargs["torch_dtype"] = (
            dtype_override if dtype_override is not None else torch.bfloat16
        )
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Dolphin model.

        Args:
            dtype_override: Optional torch.dtype to override the float input dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert float dtypes if explicitly requested (integer ids untouched).
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def decode_output(self, outputs):
        """Decode model outputs into human-readable text.

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
        """Load and return the configuration for the Dolphin model variant.

        Returns:
            The configuration object for the Dolphin model.
        """
        if self.config is None:
            self.load_model()
        return self.config
