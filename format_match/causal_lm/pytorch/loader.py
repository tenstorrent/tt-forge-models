# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FormatMatch_v1 (i1-GGUF) model loader for causal language modeling.

This loads the GGUF-quantized weights published at
``mradermacher/FormatMatch_v1-i1-GGUF``. FormatMatch_v1 is a Qwen3-4B based
tool-use / agent fine-tune; the GGUF files here are imatrix ("i1") quantizations
of that model. transformers dequantizes the selected GGUF file back into a full
``Qwen3ForCausalLM`` module (requires the ``gguf`` package).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

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
    """Available FormatMatch_v1 i1-GGUF variants for causal language modeling."""

    Q4_K_M = "q4_k_m"


class ModelLoader(ForgeModel):
    """FormatMatch_v1 (i1-GGUF) loader for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/FormatMatch_v1-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    # GGUF filename within the repo for each variant. transformers selects and
    # dequantizes this specific file via the ``gguf_file`` argument.
    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "FormatMatch_v1.i1-Q4_K_M.gguf",
    }

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language model."

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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="FormatMatch v1 (i1-GGUF)",
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
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FormatMatch_v1 GGUF model for this instance's variant.

        Args:
            dtype_override: Optional torch dtype to cast the dequantized weights to.
                            GGUF weights are dequantized to float32 by default.

        Returns:
            torch.nn.Module: The Qwen3ForCausalLM instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # GGUF loading may ignore torch_dtype and dequantize to float32; cast
        # explicitly so the model matches the requested compute dtype.
        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FormatMatch_v1 GGUF model.

        Args:
            dtype_override: Unused for integer token inputs; accepted for API parity.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
