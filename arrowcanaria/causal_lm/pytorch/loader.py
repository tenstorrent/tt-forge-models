# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ArrowCanaria-Llama model loader for causal language modeling.

This loader targets the GGUF (llama.cpp) distribution of
``mradermacher/ArrowCanaria-Llama-8B-RL-v0.1-i1-GGUF`` which is an
imatrix-quantized release of ``DataPilot/ArrowCanaria-Llama-8B-RL-v0.1``
(a Llama 3 8B fine-tune). The GGUF repository ships no ``config.json`` or
``safetensors`` weights, so both the model and the tokenizer are
reconstructed by transformers from the GGUF file via the ``gguf_file``
argument. transformers dequantizes the selected quantization back into
dense ``float32`` torch weights at load time, after which the model behaves
like an ordinary ``LlamaForCausalLM``.

The PCC validation performed on hardware compares the device output against
the CPU output of this same dequantized model, so the specific quantization
chosen only affects download size, not numerical agreement. The standard
``Q4_K_M`` quantization is used here.
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
    """Available ArrowCanaria-Llama variants (GGUF source)."""

    # 8B RL fine-tune, imatrix Q4_K_M quantization (dequantized at load time).
    ARROWCANARIA_8B_Q4_K_M = "8b_rl_q4_k_m"


class ModelLoader(ForgeModel):
    """ArrowCanaria-Llama loader for causal language modeling (GGUF source)."""

    # The GGUF file within the repository to load. transformers downloads only
    # this file and dequantizes it into dense float32 weights.
    _GGUF_FILES = {
        ModelVariant.ARROWCANARIA_8B_Q4_K_M: "ArrowCanaria-Llama-8B-RL-v0.1.i1-Q4_K_M.gguf",
    }

    _VARIANTS = {
        ModelVariant.ARROWCANARIA_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/ArrowCanaria-Llama-8B-RL-v0.1-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ARROWCANARIA_8B_Q4_K_M

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
            model="ArrowCanaria-Llama",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        The tokenizer is embedded in the GGUF file, so it must be loaded with
        the matching ``gguf_file`` argument.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Set pad token to eos token for Llama models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ArrowCanaria-Llama model instance.

        Args:
            dtype_override: Optional torch.dtype to cast the (dequantized)
                            model weights to. If not provided, the model keeps
                            the float32 weights produced by GGUF dequantization.

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
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

        # GGUF dequantization always produces float32 weights; honor the
        # requested dtype explicitly in case from_pretrained did not.
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ArrowCanaria-Llama model.

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

        # Pad input_ids and attention_mask to a fixed length
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
