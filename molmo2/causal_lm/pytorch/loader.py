# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 text-decoder (causal LM) loader.

Molmo2-8B is a custom ``trust_remote_code`` VLM (Qwen3-8B text decoder +
SigLIP-style ViT). The full ``Molmo2ForConditionalGeneration`` forward contains
``.item()`` graph breaks and data-dependent adapter gathers, so it is split into
its device-compilable components. This loader brings up the text decoder
(``model.transformer``, ``Molmo2TextModel``) plus the ``lm_head``, run logits-only
with ``use_cache=False`` so the device output is a single logits tensor
``[B, seq, vocab]`` (no KV-cache leaves to drag down the PCC gate).

See the sibling ``vision`` loader for the image tower, and ``.._compat`` for the
three transformers>=5.5 fixes both components share.
"""

from typing import Optional

import torch

from ..._compat import apply_molmo2_compat, fix_rotary_inv_freq
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
    """Available Molmo2 text-decoder variants."""

    MOLMO2_8B = "8b"


class _Molmo2TextWrapper(torch.nn.Module):
    """Wraps the text decoder + lm_head so only that subtree moves to device.

    Runs logits-only (``use_cache=False``) and returns a single logits tensor.
    """

    def __init__(self, transformer, lm_head):
        super().__init__()
        self.transformer = transformer
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask=None):
        hidden = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        return self.lm_head(hidden)


class ModelLoader(ForgeModel):
    """Loader for the Molmo2-8B text decoder (Qwen3-8B-style causal LM)."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # Natural, unpadded prompt — a single-forward bringup must not add generation
    # padding, which injects bf16 noise at masked positions (see causal-LM gate notes).
    sample_text = "The capital of France is"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name, trust_remote_code=True
            )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load the Molmo2 text decoder + lm_head wrapped for device.

        Args:
            dtype_override: Optional torch.dtype (the runner passes bfloat16).

        Returns:
            torch.nn.Module: wrapper returning logits ``[B, seq, vocab]``.
        """
        from transformers import AutoModelForImageTextToText

        # transformers>=5.5 fixes: re-register 'default' RoPE (needed at
        # construction) + neutralize the int64-cumsum packed-sequence helper that
        # the TT backend cannot legalize.
        apply_molmo2_compat()

        pretrained_model_name = self._variant_config.pretrained_model_name
        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        full = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        # The rotary inv_freq buffer is corrupted (mostly zeros) at init under
        # transformers>=5.5; recompute it or the decoder produces NaN logits.
        fix_rotary_inv_freq(full)

        model = _Molmo2TextWrapper(full.model.transformer, full.lm_head)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Tokenize the natural prompt (no padding) into ``input_ids``/``attention_mask``."""
        tokenizer = self._load_tokenizer()
        inputs = tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
