# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama (GGUF) model loader implementation for causal language modeling.

Loads GGUF-quantized Llama 3.1 checkpoints (e.g. bartowski's Lexi Uncensored V2)
via transformers' GGUF dequantization path. transformers dequantizes the GGUF
into a full fp32 state dict at load time; to avoid holding two full copies of the
weights at once (host OOM on the bringup runner), we load in fp32 and then cast
the parameters/buffers to the requested dtype in place, one tensor at a time.
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
    """Available Llama GGUF model variants for causal LM."""

    # bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF, Q4_K_M quantization
    LEXI_8B_V2_Q4_K_M = "8b_lexi_uncensored_v2_q4_k_m"


class ModelLoader(ForgeModel):
    """Llama GGUF model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LEXI_8B_V2_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
            max_length=128,
        ),
    }

    # GGUF filename within the repo for each variant. transformers selects which
    # quantized file inside the GGUF repo to dequantize via the `gguf_file` kwarg.
    _GGUF_FILES = {
        ModelVariant.LEXI_8B_V2_Q4_K_M: "Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LEXI_8B_V2_Q4_K_M

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
            model="Llama-GGUF",
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
        """Load and return the Llama GGUF model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to cast the model to. If not
                provided, the model is left in the dequantized fp32 dtype.

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        # IMPORTANT: do NOT pass torch_dtype here. transformers dequantizes the
        # GGUF into a full fp32 state dict; passing torch_dtype would make it
        # hold both the fp32 dict and a second copy in the target dtype at once
        # (~2x peak), risking host OOM on the bringup runner. Load in fp32, then
        # cast in place below so the two full copies never coexist.
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        if dtype_override is not None:
            for param in model.parameters():
                if param.data.is_floating_point():
                    param.data = param.data.to(dtype_override)
            for buf_name, buf in model.named_buffers():
                if buf.is_floating_point():
                    buf.data = buf.data.to(dtype_override)

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llama GGUF model.

        Args:
            dtype_override: Optional torch.dtype to cast floating-point inputs to.
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

    def load_config(self):
        """Load and return the configuration for the Llama GGUF model variant.

        Returns:
            The configuration object for the Llama model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )

        return self.config
