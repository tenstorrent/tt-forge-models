# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma Scope Transcoders (per-layer transcoder set) model loader implementation
for causal language modeling.

Reference: https://huggingface.co/mntss/gemma-scope-transcoders
"""

import os

import huggingface_hub.constants as _hf_constants
import torch
from transformers import AutoTokenizer
from typing import Optional

# circuit_tracer 0.5.0 imports HF_HUB_ENABLE_HF_TRANSFER which was removed in
# huggingface_hub>=1.0.0. Patch the module before circuit_tracer is imported.
if not hasattr(_hf_constants, "HF_HUB_ENABLE_HF_TRANSFER"):
    _hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False

from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Gemma Scope Transcoders model variants."""

    GEMMA_SCOPE_TRANSCODERS = "gemma-scope-transcoders"


class ModelLoader(ForgeModel):
    """Gemma Scope Transcoders model loader implementation for causal language
    modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_SCOPE_TRANSCODERS: LLMModelConfig(
            pretrained_model_name="mntss/gemma-scope-transcoders",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_SCOPE_TRANSCODERS

    BASE_MODEL = "google/gemma-2-2b"

    # transformer_lens model name for the built-in config (no HF download needed)
    _TL_MODEL_NAME = "gemma-2-2b"

    sample_text = "What is your favorite city?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Gemma-Scope-Transcoders",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _should_use_random_weights(self) -> bool:
        # Use random weights when explicitly requested or in compile-only mode,
        # since google/gemma-2-2b is a gated model requiring license acceptance.
        return os.environ.get("TT_RANDOM_WEIGHTS") == "1" or bool(
            os.environ.get("TT_COMPILE_ONLY_SYSTEM_DESC")
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer from the base Gemma model.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            tokenizer_kwargs["token"] = hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma Scope Transcoders ReplacementModel instance.

        Returns:
            The ReplacementModel wrapping the base Gemma-2-2B model with
            per-layer transcoder features from gemma-scope-transcoders.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._should_use_random_weights():
            # Use built-in transformer_lens config (no HF download needed) and
            # load transcoders from mntss/gemma-scope-transcoders (not gated).
            from circuit_tracer.replacement_model.replacement_model_transformerlens import (
                TransformerLensReplacementModel,
            )
            from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
            from transformer_lens.loading_from_pretrained import (
                get_pretrained_model_config,
            )

            cfg = get_pretrained_model_config(
                self._TL_MODEL_NAME,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
            # Skip tokenizer loading — the base model is gated, and inputs
            # are provided as pre-tokenized tensors in random-weights mode.
            cfg.tokenizer_name = None

            transcoders, _ = load_transcoder_from_hub(pretrained_model_name)
            model = TransformerLensReplacementModel.from_config(
                cfg, transcoders, **kwargs
            )
            self.model = model
            return model

        from circuit_tracer import ReplacementModel

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            kwargs.setdefault("token", hf_token)

        model = ReplacementModel.from_pretrained(
            self.BASE_MODEL, pretrained_model_name, **kwargs
        )
        self.model = model
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
    ):
        """Load and return sample inputs for the Gemma Scope Transcoders model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        max_length = self._variant_config.max_length

        if self._should_use_random_weights():
            # Return dummy token IDs without downloading the gated tokenizer.
            # HookedTransformer.forward() takes `input`, not `input_ids`.
            return {"input": torch.zeros(batch_size, max_length, dtype=torch.long)}

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_prompt = prompt or self.sample_text
        tokenized = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        # HookedTransformer.forward() takes `input`, not `input_ids`.
        input_ids = tokenized["input_ids"].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            from ...tools.utils import cast_input_to_type

            input_ids = cast_input_to_type(input_ids, dtype_override)

        return {"input": input_ids}
