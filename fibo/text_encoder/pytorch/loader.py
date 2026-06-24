# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO text-encoder loader implementation.

FIBO (briaai/FIBO) is BRIA AI's 8B-parameter DiT flow-matching text-to-image
model. Its text encoder is a ``SmolLM3ForCausalLM`` (SmolLM3-3B) hosted in the
``text_encoder`` subfolder of the gated ``briaai/FIBO`` repo. The diffusion
pipeline runs this encoder with ``output_hidden_states=True`` and feeds the
per-layer hidden states into the denoiser; for a single-component bringup we
test a plain forward pass of the encoder at the model's native token budget.

Reference: https://huggingface.co/briaai/FIBO
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
    """Available FIBO text-encoder variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """FIBO text-encoder (SmolLM3-3B) loader."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="briaai/FIBO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Native max context is 65536; a 128-token prompt is representative and
    # keeps the bringup forward tractable.
    sample_seq_len = 128
    prompt = (
        '{"subject":"a hyper-detailed, ultra-fluffy owl in moonlit trees",'
        '"style_medium":"photograph","lighting":"cool moonlight"}'
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the FIBO text-encoder loader.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIBO_text_encoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                subfolder="tokenizer",
            )
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FIBO SmolLM3 text encoder.

        Args:
            dtype_override: Optional ``torch.dtype`` for the model weights.

        Returns:
            torch.nn.Module: ``SmolLM3ForCausalLM`` in eval mode.
        """
        model_kwargs = {"subfolder": "text_encoder"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return tokenized sample inputs for the FIBO text encoder.

        Args:
            dtype_override: Unused (integer ``input_ids`` are dtype-invariant).
            batch_size: Number of identical prompts to batch.

        Returns:
            dict: ``input_ids`` and ``attention_mask`` tensors.
        """
        tokenizer = self._load_tokenizer()
        inputs = tokenizer(
            [self.prompt] * batch_size,
            padding="max_length",
            max_length=self.sample_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
