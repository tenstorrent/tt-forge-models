# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama3.1-8B-Thinking-R1 (GGUF) model loader for causal language modeling.

The weights are distributed as GGUF quantized files. Transformers dequantizes
the GGUF tensors back to a standard ``LlamaForCausalLM`` (Llama 3.1 8B
architecture) at load time, so the model runs as a normal PyTorch module.
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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Llama3.1-8B-Thinking-R1 GGUF variants."""

    # Quantization variant published in the GGUF repo.
    Q4_K_M = "8b_q4_k_m"


class ModelLoader(ForgeModel):
    """Loader for the GGUF-quantized Llama3.1-8B-Thinking-R1 causal LM."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Llama3.1-8B-Thinking-R1-GGUF",
            max_length=128,
        ),
    }

    # Maps each variant to the concrete GGUF file inside the repo.
    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "Llama3.1-8B-Thinking-R1.Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    # Sample text for causal LM.
    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with the specified variant.

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
            ModelInfo: Information about the model and variant.
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Llama3.1-8B-Thinking-R1-GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load the tokenizer (embedded in the GGUF file) for the variant.

        Returns:
            The loaded tokenizer instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Llama tokenizers ship without a pad token; reuse eos for padding.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the loaded dtype.
                            GGUF tensors are dequantized to float32 and then
                            cast to this dtype if provided.

        Returns:
            torch.nn.Module: The Llama causal LM instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this variant's settings.

        Args:
            dtype_override: Optional torch.dtype applied to the input tensors.
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
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

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # This is a forward-only (prefill) test, so no padding is needed. Feeding
        # the natural prompt length keeps every position attended (full mask),
        # which avoids the noisy logits at masked padding positions that would
        # otherwise drag down the device-vs-CPU PCC.
        self.seq_len = inputs["input_ids"].shape[-1]
        return inputs
