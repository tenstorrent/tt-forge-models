# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Altair-Stock model loader implementation for causal language modeling.

Altair-Stock-12B-v1-MPOA is a Llama-architecture (Mistral-Nemo derived) merge
published *only* in GGUF format. We load it via transformers' GGUF support
(``gguf_file=...``), which dequantizes the chosen quant to a full-precision
``LlamaForCausalLM``. The quant level only affects download size, not numerics
on device: the CPU golden and the device run share the same dequantized weights,
so a small quant is preferred.
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
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Altair-Stock model variants for causal LM."""

    ALTAIR_STOCK_12B = "12b"


# GGUF repo ships only quantized files. Use a small, standard K-quant: transformers
# dequantizes it to full precision, so this choice does not affect on-device PCC.
GGUF_FILE = "Altair-Stock-12B-v1-MPOA.i1-Q2_K.gguf"


class ModelLoader(ForgeModel):
    """Altair-Stock model loader implementation for causal LM tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.ALTAIR_STOCK_12B: LLMModelConfig(
            pretrained_model_name="mradermacher/Altair-Stock-12B-v1-MPOA-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ALTAIR_STOCK_12B

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
            model="Altair-Stock",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant (sourced from the GGUF file).

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=GGUF_FILE
        )

        # Set pad token to eos token (Llama-family models have no dedicated pad).
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Altair-Stock model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its dequantized dtype (float32).

        Returns:
            torch.nn.Module: The model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

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
            dict: Input tensors suitable for causal LM.
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

        # Only convert dtype if explicitly requested (casts float tensors only).
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask to a fixed length.
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_config(self):
        """Load and return the configuration for the Altair-Stock model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=GGUF_FILE
        )

        return self.config
