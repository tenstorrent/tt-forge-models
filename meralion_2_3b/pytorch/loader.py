# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MERaLiON-2-3B model loader implementation.

MERaLiON-2-3B is a multimodal speech-text large language model from A*STAR
tailored for Singapore's multilingual landscape. It integrates a localized
Whisper-Large-V3 speech encoder with a Gemma2-2b-IT text decoder to support
automatic speech recognition (ASR), speech translation, audio captioning,
and audio question answering across English, Mandarin, Malay, Tamil,
Indonesian, Thai, and Vietnamese.
"""

from typing import Optional

import numpy as np
import torch
import transformers.cache_utils as _cache_utils
import transformers.modeling_utils as _modeling_utils
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# HybridCache was removed in transformers 5.x; inject a stub so the model's
# trust_remote_code module (which does an isinstance check against it) can import.
if not hasattr(_cache_utils, "HybridCache"):
    from transformers.cache_utils import Cache

    class HybridCache(Cache):
        pass

    _cache_utils.HybridCache = HybridCache

# The model defines _supports_sdpa as a property accessing self.text_decoder, but
# PreTrainedModel.__init__ checks _supports_sdpa before text_decoder is initialized,
# raising AttributeError. Wrap _sdpa_can_dispatch to treat that as "not supported".
_orig_sdpa_can_dispatch = _modeling_utils.PreTrainedModel._sdpa_can_dispatch


def _safe_sdpa_can_dispatch(self, is_init_check=False):
    try:
        return _orig_sdpa_can_dispatch(self, is_init_check)
    except AttributeError:
        return False


_modeling_utils.PreTrainedModel._sdpa_can_dispatch = _safe_sdpa_can_dispatch

# Custom models that override tie_weights() without the recompute_mapping kwarg added
# in transformers 5.x cause a TypeError when init_weights() calls tie_weights(recompute_mapping=False).
# Wrap init_weights to temporarily patch the class's tie_weights so it accepts the kwarg.
_orig_init_weights = _modeling_utils.PreTrainedModel.init_weights


def _compat_init_weights(self):
    try:
        return _orig_init_weights(self)
    except TypeError as e:
        if "recompute_mapping" not in str(e):
            raise
        orig_tie_weights = type(self).tie_weights
        type(self).tie_weights = lambda self_inner, **kwargs: orig_tie_weights(
            self_inner
        )
        try:
            return _orig_init_weights(self)
        finally:
            type(self).tie_weights = orig_tie_weights


_modeling_utils.PreTrainedModel.init_weights = _compat_init_weights

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MERaLiON-2-3B model variants."""

    V3B = "3B"


class MERaLiON2Wrapper(torch.nn.Module):
    """Wrapper around MERaLiON2ForConditionalGeneration for a clean forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, attention_mask, input_features, feature_attention_mask
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )


class ModelLoader(ForgeModel):
    """MERaLiON-2-3B model loader implementation."""

    _VARIANTS = {
        ModelVariant.V3B: ModelConfig(
            pretrained_model_name="MERaLiON/MERaLiON-2-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V3B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MERaLiON_2_3B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MERaLiON-2-3B model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "use_safetensors": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        # Load config first so we can back-fill token IDs removed in transformers 5.x
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        for _attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
            if not hasattr(model_config, _attr):
                setattr(model_config, _attr, None)

        # Pre-import the model class so we can patch tie_weights before from_pretrained
        # calls it. The custom model's tie_weights(self) doesn't accept the
        # missing_keys / recompute_mapping kwargs added in transformers 5.x.
        _model_cls = get_class_from_dynamic_module(
            "modeling_meralion2.MERaLiON2ForConditionalGeneration",
            pretrained_model_name,
            trust_remote_code=True,
        )
        _orig_tie_weights = _model_cls.__dict__.get("tie_weights")
        if _orig_tie_weights is not None:
            _model_cls.tie_weights = lambda self, **kwargs: _orig_tie_weights(self)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name,
            config=model_config,
            **model_kwargs,
        )
        model.eval()

        self._processor = AutoProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        return MERaLiON2Wrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MERaLiON-2-3B model."""
        if self._processor is None:
            self.load_model(dtype_override=dtype_override)

        prompt_template = (
            "Instruction: {query} \n"
            "Follow the text instruction based on the following audio: <SpeechHere>"
        )
        conversation = [
            [
                {
                    "role": "user",
                    "content": prompt_template.format(
                        query="Please transcribe this speech."
                    ),
                }
            ],
        ]
        chat_prompt = self._processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate a synthetic 1-second mono audio clip at 16 kHz
        audio_array = [np.random.randn(16000).astype(np.float32)]

        inputs = self._processor(text=chat_prompt, audios=audio_array)

        input_features = inputs["input_features"]
        if dtype_override is not None and input_features.is_floating_point():
            input_features = input_features.to(dtype_override)

        return [
            inputs["input_ids"],
            inputs["attention_mask"],
            input_features,
            inputs["feature_attention_mask"],
        ]
