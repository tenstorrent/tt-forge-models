# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
A0l-12B-heretic model loader implementation for causal language modeling.

This loads a llama.cpp GGUF export (mradermacher/A0l-12B-heretic-i1-GGUF).
HuggingFace transformers de-quantizes the GGUF weights back into a normal
torch causal-LM module via the ``gguf_file=`` argument, so the loaded model
and tokenizer behave like any other HF causal LM. The base model is
nbeerbower/A0l-12B-heretic (a 12B Mistral-Nemo derivative); the GGUF declares
``general.architecture=llama`` so transformers materializes it as a
``LlamaForCausalLM`` module.
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


class ModelVariant(StrEnum):
    """Available A0l-12B-heretic model variants for causal LM."""

    HERETIC_12B_I1 = "12B_heretic_i1"


class ModelLoader(ForgeModel):
    """A0l-12B-heretic GGUF model loader for causal language modeling tasks."""

    # GGUF quant file to de-quantize. Any quant gives identical de-quantized
    # weights (and thus PCC); Q4_K_M is a k-quant well supported by the
    # transformers GGUF dequantizer.
    _GGUF_FILE = "A0l-12B-heretic.i1-Q4_K_M.gguf"

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.HERETIC_12B_I1: LLMModelConfig(
            pretrained_model_name="mradermacher/A0l-12B-heretic-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.HERETIC_12B_I1

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
            model="A0l-12B-heretic",
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._GGUF_FILE
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the A0l-12B-heretic model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. If not provided, the model uses its default.

        Returns:
            torch.nn.Module: The model instance for causal language modeling.
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
        """Load and return sample inputs for the model with this variant's settings.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
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
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
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
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._GGUF_FILE
        )
        return self.config
