# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EstopianMaid-13B model loader implementation for causal language modeling.

EstopianMaid-13B is a Llama-2 13B based merge distributed only in GGUF format.
The loader uses transformers' GGUF support (``gguf_file=``) to dequantize the
quantized weights into a standard ``LlamaForCausalLM`` instance at load time.
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
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available EstopianMaid model variants for causal LM."""

    ESTOPIANMAID_13B = "13B"


class ModelLoader(ForgeModel):
    """EstopianMaid model loader implementation for causal language modeling tasks."""

    # GGUF file (within the HF repo) to dequantize. Q4_K_S is the smaller of the
    # two published quantizations.
    _GGUF_FILE = "EstopianMaid-13B-Q4_K_S.gguf"

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.ESTOPIANMAID_13B: LLMModelConfig(
            pretrained_model_name="KatyTheCutie/EstopianMaid-13B-GGUF",
            max_length=32,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ESTOPIANMAID_13B

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
        self.seq_len = None
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
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="EstopianMaid",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        The tokenizer is reconstructed from the embedded vocabulary in the GGUF
        file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._GGUF_FILE
        )

        # Set pad token to eos token (Llama family has no dedicated pad token).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the EstopianMaid model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the dequantized dtype.
                            If not provided, the model uses its default dtype.

        Returns:
            torch.nn.Module: The EstopianMaid (LlamaForCausalLM) model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the EstopianMaid model.

        Args:
            dtype_override: Optional torch.dtype to override the default input dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # For causal LM, we need both input_ids and attention_mask
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_config(self):
        """Load and return the configuration for the EstopianMaid model variant.

        The config is embedded in the GGUF file, so this defers to model load.

        Returns:
            The configuration object for the model, or None if not yet loaded.
        """
        return self.config
