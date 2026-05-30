# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Aura v2 7B model loader implementation for causal language modeling.

Aura_v2_7B is a Mistral-7B based, LLaVA-style multimodal merge distributed only
as GGUF weights. There is no safetensors / config.json checkpoint, so both the
config and tokenizer are reconstructed from the GGUF metadata via the
``gguf_file`` argument to ``from_pretrained`` (requires the ``gguf`` package).

Only the text (language model) tower is exercised here: the vision tower is
shipped separately as ``mmproj-model-f16.gguf`` and is intentionally not loaded.
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


class ModelVariant(StrEnum):
    """Available Aura model variants for causal LM."""

    V2_7B = "v2_7b"


class ModelLoader(ForgeModel):
    """Aura v2 7B model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.V2_7B: LLMModelConfig(
            pretrained_model_name="Lewdiculous/Aura_v2_7B-GGUF-IQ-Imatrix",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.V2_7B

    # GGUF file (within the HF repo) used to reconstruct each variant's weights,
    # config and tokenizer. A K-quant is used because transformers' GGUF loader
    # does not support the IQ imatrix quant types this repo also ships.
    _GGUF_FILES = {
        ModelVariant.V2_7B: "Aura_v2_7B-Q4_K_M-imat.gguf",
    }

    # Longer sample text (~50 tokens) for causal LM. Short sequences are known to
    # degrade PCC for large LLMs, so an unpadded real prompt is used.
    sample_text = (
        "The history of artificial intelligence spans many decades, from early "
        "symbolic reasoning systems to modern large language models trained on "
        "vast text corpora. Researchers continue to explore how these models "
        "reason, generalize, and represent knowledge about the world around us."
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
        return ModelInfo(
            model="Aura",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        """GGUF filename within the HF repo for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from the GGUF metadata.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )

        # Set pad token to eos token for Mistral-style models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Aura model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The Aura model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        # Default to bfloat16 unless explicitly overridden.
        model_kwargs["torch_dtype"] = (
            dtype_override if dtype_override is not None else torch.bfloat16
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
        """Load and return sample inputs for the Aura model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # For causal LM, we need both input_ids and attention_mask.
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        self.seq_len = inputs["input_ids"].shape[1]

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs):
        """Decode model outputs into the next-token text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded next token text
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        """Load and return the configuration for the Aura model variant.

        Returns:
            The configuration object reconstructed from the GGUF metadata.
        """
        if self.config is None:
            # The config is only available through the GGUF metadata, which is
            # parsed when the model is loaded.
            self.load_model()
        return self.config
