# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
L3-8B-Stheno-v3.2 (GGUF) model loader implementation for causal language modeling.

The HuggingFace repo ``Den4iiks/L3-8B-Stheno-v3.2-GGUF-IQ-Imatrix`` ships only
quantized GGUF (llama.cpp) weight files, with no config.json/safetensors at the
repo root. transformers can still load these by passing ``gguf_file=<path>`` to
``from_pretrained``: it parses the GGUF metadata to rebuild the HF config and
tokenizer and de-quantizes the weights to float32 at load time. The de-quantized
float32 model is then cast to the requested ``dtype_override`` (the test harness
passes ``bfloat16``).

The underlying architecture is Llama 3 8B (``LlamaForCausalLM``); the base model
is ``Sao10K/L3-8B-Stheno-v3.2``.
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
    """Available L3-8B-Stheno GGUF model variants for causal LM."""

    Q4_K_M = "q4_k_m"


class ModelLoader(ForgeModel):
    """L3-8B-Stheno (GGUF) loader implementation for causal language modeling tasks."""

    _HF_REPO = "Den4iiks/L3-8B-Stheno-v3.2-GGUF-IQ-Imatrix"

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.Q4_K_M: LLMModelConfig(
            pretrained_model_name=_HF_REPO,
            max_length=128,
        ),
    }

    # In-repo GGUF file selected per variant. transformers de-quantizes these to fp32.
    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "L3-8B-Stheno-v3.2-Q4_K_M-imat.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

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
            model="L3-8B-Stheno-v3.2",
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

        # Set pad token to eos token for Llama models
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the L3-8B-Stheno model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to cast the (de-quantized fp32)
                model to. If not provided, the model stays in float32.

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        # Loading via gguf_file de-quantizes the weights to float32.
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype. Only applied to floating-point
                tensors; integer token ids are left unchanged.
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

        # Only convert dtype if explicitly requested (no-op for integer ids)
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
