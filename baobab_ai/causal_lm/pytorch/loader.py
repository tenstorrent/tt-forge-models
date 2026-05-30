# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BaobabAI model loader implementation for causal language modeling.

BaobabAI-v0.2 is a Llama-3.2-3B fine-tune for African languages. This loader
consumes the imatrix-quantized GGUF distribution published by ``mradermacher``
(``mradermacher/BaobabAI-v0.2-i1-GGUF``). The GGUF declares the ``llama``
architecture, so transformers can natively dequantize it to a standard
``LlamaForCausalLM`` via the ``gguf_file`` argument (requires the ``gguf``
package, declared in requirements.txt).
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
    """Available BaobabAI model variants for causal LM."""

    # imatrix-quantized GGUF, Q4_K_M (4-bit K-quant, mixed) — dequantized to a
    # standard LlamaForCausalLM at load time.
    V0_2_I1_Q4_K_M_GGUF = "v0.2_i1_Q4_K_M_gguf"


# GGUF filename within the repo for each variant.
_GGUF_FILES = {
    ModelVariant.V0_2_I1_Q4_K_M_GGUF: "BaobabAI-v0.2.i1-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """BaobabAI model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.V0_2_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/BaobabAI-v0.2-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.V0_2_I1_Q4_K_M_GGUF

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
            model="BaobabAI",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        """GGUF filename within the repo for the current variant."""
        return _GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF distribution.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )

        # Set pad token to eos token for Llama models
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the BaobabAI model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The BaobabAI (LlamaForCausalLM) model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # transformers dequantizes the GGUF tensors and builds a standard
        # LlamaForCausalLM (the GGUF declares the llama architecture).
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the BaobabAI model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
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
