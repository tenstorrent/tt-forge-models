# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Azure-Starlight-12B-Heretic model loader for causal language modeling.

This is a GGUF-quantized community model (mradermacher/Azure-Starlight-12B-Heretic-i1-GGUF),
a fine-tune of the Mistral-Nemo 12B architecture. transformers reconstructs the
config, tokenizer and (dequantized) weights from the selected GGUF file via the
``gguf_file`` argument. The GGUF "llama" architecture maps to LlamaForCausalLM in
transformers, with a separate ``head_dim`` (128) distinct from hidden_size/num_heads.
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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Azure-Starlight model variants for causal LM."""

    # i1 (imatrix) Q4_K_M GGUF quantization of the 12B Heretic model.
    HERETIC_12B_I1_Q4_K_M = "12b_heretic_i1_q4_k_m"


class ModelLoader(ForgeModel):
    """Azure-Starlight-12B-Heretic loader for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.HERETIC_12B_I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Azure-Starlight-12B-Heretic-i1-GGUF",
            max_length=128,
        ),
    }

    # GGUF file within the repo to load for each variant.
    _GGUF_FILES = {
        ModelVariant.HERETIC_12B_I1_Q4_K_M: "Azure-Starlight-12B-Heretic.i1-Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.HERETIC_12B_I1_Q4_K_M

    # Sample text for causal LM, long enough to fill the sequence with real
    # tokens (minimal padding keeps the PCC comparison focused on real content).
    sample_text = (
        "The history of artificial intelligence spans many decades, beginning "
        "with early symbolic systems and evolving toward large neural networks "
        "that can understand and generate human language with remarkable fluency."
    )

    # Fixed, tile-aligned sequence length for the forward pass.
    seq_length = 32

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
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
            model="Azure-Starlight-12B-Heretic",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self):
        """Return the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )

        # Set pad token to eos token (GGUF llama tokenizers have no pad token).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Azure-Starlight model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the dequantized weight
                            dtype. If not provided, defaults to float32.

        Returns:
            torch.nn.Module: The model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        # Keep GGUF's native fp32 dequantization by default (the device path is
        # precision-invariant here, so this mainly fixes the host-side golden).
        model_kwargs["torch_dtype"] = (
            dtype_override if dtype_override is not None else torch.float32
        )
        model_kwargs |= kwargs

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
            dtype_override: Optional torch.dtype to override input dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Tokenize to a fixed, tile-aligned length with minimal padding so the
        # PCC comparison is dominated by real-token activations rather than pad.
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.seq_length,
        )
        self.seq_len = self.seq_length

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def load_config(self):
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )
        return self.config
