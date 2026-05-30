# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama (GGUF) model loader implementation for causal language modeling.

Loads a GGUF-quantized Llama checkpoint published on the HuggingFace Hub. The
GGUF file is self-contained: transformers reads the architecture config and
tokenizer directly from the GGUF metadata and dequantizes the weights into a
standard torch ``LlamaForCausalLM`` module, so the (gated) base model repo is
never required.

Default checkpoint: bartowski's GGUF build of
SicariusSicariiStuff/Llama-3.3-8B-Instruct-128K_Abliterated.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Llama GGUF model variants for causal LM."""

    LLAMA_3_3_8B_ABLITERATED_Q4_K_M = "3.3_8B_Abliterated_Q4_K_M"


class ModelLoader(ForgeModel):
    """Loader for GGUF-quantized Llama causal LM checkpoints."""

    # All variants below live in the same GGUF repo; only the file differs.
    _GGUF_REPO = (
        "bartowski/SicariusSicariiStuff_Llama-3.3-8B-Instruct-128K_Abliterated-GGUF"
    )

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LLAMA_3_3_8B_ABLITERATED_Q4_K_M: LLMModelConfig(
            pretrained_model_name=_GGUF_REPO,
            max_length=128,
        ),
    }

    # Maps each variant to the GGUF file to dequantize from the repo above.
    _GGUF_FILES = {
        ModelVariant.LLAMA_3_3_8B_ABLITERATED_Q4_K_M: (
            "SicariusSicariiStuff_Llama-3.3-8B-Instruct-128K_Abliterated-Q4_K_M.gguf"
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLAMA_3_3_8B_ABLITERATED_Q4_K_M

    # Sample text for causal LM. Kept reasonably long (~40 real tokens, unpadded)
    # because Llama PCC on Blackhole is sensitive to very short input sequences.
    sample_text = (
        "The history of artificial intelligence began in antiquity, with myths "
        "and stories about artificial beings endowed with intelligence by master "
        "craftsmen. Modern research into the field formally began in the middle "
        "of the twentieth century, and has advanced rapidly ever since."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

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
            model="LlamaGGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        """Return the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF metadata.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )

        # Llama checkpoints ship without a dedicated pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Llama GGUF model instance for this variant.

        transformers dequantizes the GGUF weights into a standard
        ``LlamaForCausalLM`` module.

        Args:
            dtype_override: Optional torch dtype for the dequantized weights.
                            If not provided, transformers uses float32.

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # The GGUF dequantization path may not honor torch_dtype for every
        # tensor, so make the requested dtype authoritative.
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llama GGUF model.

        Args:
            dtype_override: Unused for token inputs (ids/masks stay integer),
                            accepted for interface compatibility.
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors suitable for causal LM (input_ids, attention_mask).
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for the requested batch size.
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
