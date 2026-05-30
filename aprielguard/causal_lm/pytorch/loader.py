# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AprielGuard model loader implementation for causal language modeling.

AprielGuard (ServiceNow-AI/AprielGuard) is published only as GGUF (llama.cpp)
quantizations under ``mradermacher/AprielGuard-i1-GGUF``. The base model is a
Mistral-architecture causal LM; in GGUF form llama.cpp stores it under the
``llama`` architecture, so transformers materializes it as ``LlamaForCausalLM``
after de-quantizing the GGUF tensors back to the requested dtype. The weights,
tokenizer and config are all read from the chosen ``.gguf`` file via the
``gguf_file`` argument to ``from_pretrained``.
"""

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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
    """Available AprielGuard model variants for causal LM."""

    # imatrix (i1) Q4_K_M GGUF quantization. transformers de-quantizes the
    # K-quant weights back to the requested float dtype on load.
    I1_Q4_K_M = "i1_q4_k_m"


class ModelLoader(ForgeModel):
    """AprielGuard model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/AprielGuard-i1-GGUF",
            max_length=128,
        ),
    }

    # GGUF weight file (within the HF repo above) backing each variant.
    _GGUF_FILES = {
        ModelVariant.I1_Q4_K_M: "AprielGuard.i1-Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.I1_Q4_K_M

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
            model="AprielGuard",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self) -> str:
        """GGUF filename backing the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file
        )

        # Ensure a pad token exists for batched/padded tokenization.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AprielGuard model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to cast the de-quantized GGUF
                weights to. If not provided, float32 is used (safe on CPU).

        Returns:
            torch.nn.Module: The AprielGuard model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        # GGUF tensors are de-quantized to this dtype on load.
        dtype = dtype_override if dtype_override is not None else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            gguf_file=self._gguf_file,
            dtype=dtype,
            **kwargs,
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the AprielGuard model.

        Args:
            dtype_override: Unused for causal LM inputs (token ids stay integer);
                accepted for interface compatibility.
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
        """Decode model outputs into the predicted next token text.

        Args:
            outputs: Model output from a forward pass.

        Returns:
            str: Decoded next token text.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token = logits[:, -1].softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        """Load and return the configuration for the AprielGuard variant.

        Returns:
            The configuration object read from the GGUF file.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
