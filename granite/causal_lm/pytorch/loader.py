# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite Code model loader implementation for causal language modeling.

Weights are sourced from a GGUF repository; transformers dequantizes the GGUF
tensors back to full precision on load (requires the ``gguf`` package). The
underlying architecture is Llama-family.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    """Available Granite Code model variants for causal LM."""

    GRANITE_8B_CODE_INSTRUCT_128K = "8b_code_instruct_128k"


class ModelLoader(ForgeModel):
    """Granite Code model loader implementation for causal language modeling tasks."""

    # GGUF quantization file to dequantize from. Quant level does not affect
    # device-vs-host PCC (both run the same dequantized weights); a mid-size
    # K-quant keeps the download/memory footprint modest.
    _GGUF_FILES = {
        ModelVariant.GRANITE_8B_CODE_INSTRUCT_128K: "granite-8b-code-instruct-128k.Q4_K_M.gguf",
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GRANITE_8B_CODE_INSTRUCT_128K: LLMModelConfig(
            pretrained_model_name="RichardErkhov/ibm-granite_-_granite-8b-code-instruct-128k-gguf",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GRANITE_8B_CODE_INSTRUCT_128K

    # Sample text for causal LM
    sample_text = "def fibonacci(n):"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
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
            model="Granite",
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
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Ensure a pad token is defined for batching.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Granite model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to cast the model to. If not
                provided, the model keeps the dequantized GGUF dtype (float32).

        Returns:
            torch.nn.Module: The Granite model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # GGUF dequantization may ignore torch_dtype for some tensors; ensure the
        # whole model is in the requested dtype.
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Granite model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to cast float input tensors to.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Feed the natural (fully-attended) prompt length. A single forward pass
        # does not require padding, and padded positions hurt whole-tensor PCC.
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested (affects float tensors only)
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def load_config(self):
        """Load and return the configuration for the Granite model variant.

        Returns:
            The configuration object for the Granite model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
