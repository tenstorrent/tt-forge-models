# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dolphin (Llama 3) GGUF model loader implementation for causal language modeling.

dolphin-2.9-llama3-8b is a Llama-3 8B fine-tune distributed in GGUF format by
QuantFactory. transformers loads and dequantizes the GGUF weights on the fly via
the ``gguf_file`` argument to ``from_pretrained``.
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
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
)


class ModelVariant(StrEnum):
    """Available Dolphin model variants for causal LM."""

    DOLPHIN_2_9_LLAMA3_8B = "2_9_llama3_8b"


class ModelLoader(ForgeModel):
    """Dolphin (Llama 3) GGUF model loader for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DOLPHIN_2_9_LLAMA3_8B: LLMModelConfig(
            pretrained_model_name="QuantFactory/dolphin-2.9-llama3-8b-GGUF",
            max_length=128,
        ),
    }

    # GGUF file (within the repo) to load and dequantize for each variant.
    _GGUF_FILES = {
        ModelVariant.DOLPHIN_2_9_LLAMA3_8B: "dolphin-2.9-llama3-8b.Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DOLPHIN_2_9_LLAMA3_8B

    # Sample text for causal LM. A longer prompt keeps the real-token fraction
    # high after padding (see load_inputs), which keeps device PCC above the
    # 0.99 threshold — a short prompt left the sequence mostly zero-padding and
    # the padded positions dominated the (low) PCC.
    sample_text = (
        "The capital of France is Paris, a city known for its art, fashion, "
        "gastronomy and culture. The river Seine runs through it."
    )

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
            model="Dolphin",
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
        """Load and return the Dolphin model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its default dtype.

        Returns:
            torch.nn.Module: The Dolphin model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
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
        """Load and return sample inputs for the Dolphin model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to cast input tensors to.
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

        # Pad up to the next multiple of 32 (a tile boundary) only. pad_inputs
        # *adds* the second arg to the real length, so we pass just the number
        # of pad tokens needed to reach the tile boundary — this keeps the
        # padding fraction small so device PCC stays above threshold.
        real_len = inputs["input_ids"].shape[1]
        pad_amount = (-real_len) % 32
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], pad_amount)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], pad_amount)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_config(self):
        """Load and return the configuration for the Dolphin model variant.

        Returns:
            The configuration object for the Dolphin model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )

        return self.config
