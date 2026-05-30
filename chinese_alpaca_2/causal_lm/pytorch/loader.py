# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chinese-Alpaca-2 model loader implementation for causal language modeling.

The upstream repo ``hfl/chinese-alpaca-2-7b-gguf`` only ships GGUF weight files
(no HuggingFace-format safetensors / config.json). transformers can load these
directly via the ``gguf_file`` argument, which dequantizes the weights into a
standard ``LlamaForCausalLM`` (Chinese-Alpaca-2 is a Llama-2 architecture) and
reconstructs the tokenizer from the GGUF metadata.
"""

from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Available Chinese-Alpaca-2 model variants for causal LM."""

    CHINESE_ALPACA_2_7B = "7b"


class ModelLoader(ForgeModel):
    """Chinese-Alpaca-2 model loader for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.CHINESE_ALPACA_2_7B: LLMModelConfig(
            pretrained_model_name="hfl/chinese-alpaca-2-7b-gguf",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CHINESE_ALPACA_2_7B

    # GGUF weight file (within the repo) to load for each variant. The repo ships
    # several quantizations; the f16 file is the highest-fidelity representation.
    _GGUF_FILE = {
        ModelVariant.CHINESE_ALPACA_2_7B: "ggml-model-f16.gguf",
    }

    # Sample text for causal LM (Chinese-Alpaca-2 is a Chinese/English bilingual model)
    sample_text = "请介绍一下中国的首都北京。"

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
        return ModelInfo(
            model="Chinese-Alpaca-2",
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
        gguf_file = self._GGUF_FILE[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Set pad token to eos token for Llama-family models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Chinese-Alpaca-2 model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.
                            Defaults to torch.bfloat16 so the 7B model fits on a
                            single device (GGUF weights are dequantized to fp32
                            otherwise, which is ~28GB).

        Returns:
            torch.nn.Module: The Chinese-Alpaca-2 model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILE[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        # Default to bfloat16 so the dequantized 7B weights fit on a single chip.
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        model_kwargs = {"gguf_file": gguf_file, "torch_dtype": dtype}
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Chinese-Alpaca-2 model.

        Args:
            dtype_override: Optional torch.dtype to cast float inputs to.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) suitable for causal LM.
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

        # Only convert dtype if explicitly requested (float tensors only)
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask to a fixed length for static shapes
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_config(self):
        """Load and return the configuration for the Chinese-Alpaca-2 model.

        The config is derived from the GGUF file, so it is populated by
        ``load_model``. This is a convenience accessor that loads the model if
        needed.

        Returns:
            The configuration object for the model.
        """
        if self.config is None:
            self.load_model()
        return self.config
