# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Molmo2-8B text-decoder loader (multimodal LM backbone, causal LM path).

allenai/Molmo2-8B is a custom (`trust_remote_code`) VLM: a Qwen3-style text
decoder (``Molmo2TextModel``) plus a SigLIP-style vision tower. The full
``Molmo2ForConditionalGeneration`` forward contains ``.item()`` graph breaks and
data-dependent adapter gathers, so — like other VLMs here — it is brought up as
two separate components. This loader is the **text decoder**: ``input_ids`` ->
logits over the text decoder + ``lm_head``. The vision tower lives in
``molmo2/vision/pytorch/loader.py``.

See ``molmo2/_compat.py`` for the three transformers>=5.5 fixes this needs.
"""

from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

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
from ..._compat import apply_all, register_default_rope

# Pinned revision: the custom modeling code is versioned with the weights.
_MOLMO2_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"


class ModelVariant(StrEnum):
    """Available Molmo2 text-decoder variants."""

    MOLMO2_8B = "8b"


class _Molmo2TextDecoder(torch.nn.Module):
    """Text-only forward: input_ids -> logits, no vision tower, no KV cache.

    Wraps the loaded full model's text transformer + lm_head so only the decoder
    (not the vision backbone) is compiled and moved to device.
    """

    def __init__(self, full_model):
        super().__init__()
        self.transformer = full_model.model.transformer
        self.lm_head = full_model.lm_head

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        return self.lm_head(hidden_states)


class ModelLoader(ForgeModel):
    """Loader for the Molmo2-8B text decoder (causal LM path)."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # A natural, unpadded prompt — a single-forward bringup feeds the real prompt
    # (no generation padding) to keep the PCC gate measuring the decoder, not
    # masked, never-read positions.
    sample_text = "The Tenstorrent Blackhole processor is designed to"

    def __init__(self, variant: Optional[ModelVariant] = None):
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
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                revision=_MOLMO2_REVISION,
                trust_remote_code=True,
            )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load the Molmo2 text decoder.

        Loads the full custom model (so the component modules are wired exactly
        as trained), applies the transformers>=5.5 compat fixes, then returns a
        text-only wrapper (decoder + lm_head).
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # fix #1 must be in place before the model (and its RoPE) is constructed.
        register_default_rope()

        model_kwargs = {"trust_remote_code": True, "revision": _MOLMO2_REVISION}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        full_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        full_model.eval()

        # fix #2 (recompute inv_freq) + fix #3 (disable packed-seq int64 cumsum).
        apply_all(text_model=full_model.model.transformer)

        wrapper = _Molmo2TextDecoder(full_model).eval()
        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Tokenize the natural prompt; returns input_ids + attention_mask."""
        tokenizer = self._load_tokenizer()
        inputs = tokenizer(self.sample_text, return_tensors="pt")

        encoded = {
            "input_ids": inputs["input_ids"].repeat_interleave(batch_size, dim=0),
            "attention_mask": inputs["attention_mask"].repeat_interleave(
                batch_size, dim=0
            ),
        }
        return encoded
