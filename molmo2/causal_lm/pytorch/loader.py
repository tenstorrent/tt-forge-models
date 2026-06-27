# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Molmo2 text-decoder loader (Qwen3-8B style decoder).

Brings up the language-model backbone of ``allenai/Molmo2-8B`` as a single
logits-only forward pass. The full ``Molmo2ForConditionalGeneration`` graph has
``.item()`` graph breaks and data-dependent adapter gathers, so we extract the
text decoder (``model.model.transformer`` -> ``Molmo2TextModel``) and the LM head
(``model.lm_head``) and bring them up separately from the vision tower.

The decoder is run with ``use_cache=False`` and a natural (unpadded) prompt so
the runner's PCC gate measures logits only, not KV-cache leaves or masked
padding positions.
"""

from typing import Optional

import torch

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
from ..._compat import (
    register_default_rope,
    fix_rotary_inv_freq,
    patch_packed_sequence_indices,
)


class ModelVariant(StrEnum):
    """Available Molmo2 text-decoder variants."""

    MOLMO2_8B = "8b"


class _TextDecoder(torch.nn.Module):
    """Decoder + LM head -> logits only (no KV cache, no past_key_values)."""

    def __init__(self, text_model: torch.nn.Module, lm_head: torch.nn.Module):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask=None):
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self.lm_head(outputs.last_hidden_state)


class ModelLoader(ForgeModel):
    """Molmo2 text-decoder loader."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    sample_text = "Describe what you see in a photograph of a city street at night."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load the Molmo2 text decoder + LM head for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to load weights in (e.g. bfloat16).

        Returns:
            torch.nn.Module: The wrapped decoder producing logits.
        """
        from transformers import AutoModelForImageTextToText

        pretrained_model_name = self._variant_config.pretrained_model_name

        # transformers >= 5.5 compatibility (see molmo2/_compat.py).
        register_default_rope()
        patch_packed_sequence_indices()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        full_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        full_model.eval()
        self.config = full_model.config

        text_model = full_model.model.transformer  # Molmo2TextModel
        # Non-persistent inv_freq buffers are zeroed by from_pretrained's meta
        # materialization; recompute them so RoPE doesn't emit NaNs.
        fix_rotary_inv_freq(text_model)

        decoder = _TextDecoder(text_model, full_model.lm_head)
        decoder.eval()
        if dtype_override is not None:
            decoder = decoder.to(dtype_override)
        return decoder

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Tokenize a natural (unpadded) prompt.

        Args:
            dtype_override: Unused for ids (kept for interface symmetry).
            batch_size: Batch size for the inputs.

        Returns:
            dict: {"input_ids": ..., "attention_mask": ...}.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
