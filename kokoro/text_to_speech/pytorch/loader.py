# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is an 82M-parameter open-weight TTS model built on the StyleTTS 2 +
ISTFTNet architecture (https://huggingface.co/hexgrad/Kokoro-82M). It is a
multi-stage pipeline rather than a single transformer:

    input_ids -> PL-BERT (CustomAlbert) text encoder
              -> prosody predictor (LSTM duration / F0 / energy)
              -> duration-based alignment expansion (data-dependent length)
              -> text encoder + ISTFTNet decoder -> audio waveform

``ModelLoader`` exposes the full pipeline via ``forward_with_tokens`` so the
loader is a faithful end-to-end Kokoro loader (operates on phoneme token ids and
a voice/style reference vector, skipping the misaki/espeak G2P front-end so no
external phonemizer is required at load time).
"""
from typing import Optional

import torch
from torch import nn

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Kokoro model variants."""

    BASE = "82M"


class _KokoroForward(nn.Module):
    """Thin tensor-in wrapper around ``KModel.forward_with_tokens``.

    The public ``KModel.forward`` takes a phoneme *string* and runs G2P; the
    internal ``forward_with_tokens`` takes already-tokenized phoneme ids plus a
    voice/style reference and runs the full acoustic pipeline. We wrap the latter
    so the standard ``model(**inputs)`` contract works with plain tensors.
    """

    def __init__(self, kmodel: nn.Module):
        super().__init__()
        self.kmodel = kmodel

    def forward(self, input_ids, ref_s, speed: float = 1.0):
        audio, _pred_dur = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return audio


class ModelLoader(ForgeModel):
    """Kokoro-82M model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # A fixed phoneme string (IPA) used to build deterministic sample inputs.
    # These are valid Kokoro vocab phonemes for an English utterance.
    DEFAULT_PHONEMES = "hˈɛlˌoʊ wˈɜːld"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._kmodel = None
        self._input_ids = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="kokoro",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load the full Kokoro KModel and wrap it for tensor-in forward.

        ``disable_complex=True`` selects the real-valued STFT path in the
        ISTFTNet decoder (avoids ``torch.view_as_complex`` / complex iSTFT).
        """
        from kokoro import KModel

        model_name = self._variant_config.pretrained_model_name
        kmodel = KModel(repo_id=model_name, disable_complex=True).eval()
        if dtype_override is not None:
            kmodel = kmodel.to(dtype_override)
        self._kmodel = kmodel

        model = _KokoroForward(kmodel).eval()
        return model

    def _build_input_ids(self):
        if self._kmodel is None:
            self.load_model()
        ids = [self._kmodel.vocab.get(p) for p in self.DEFAULT_PHONEMES]
        ids = [i for i in ids if i is not None]
        # KModel wraps the sequence with leading/trailing 0 (boundary) tokens.
        input_ids = torch.LongTensor([[0, *ids, 0]])
        self._input_ids = input_ids
        return input_ids

    def load_inputs(self, dtype_override=None):
        """Return sample inputs: phoneme token ids + a real voice/style vector."""
        from huggingface_hub import hf_hub_download

        input_ids = self._build_input_ids()

        model_name = self._variant_config.pretrained_model_name
        voice_path = hf_hub_download(model_name, "voices/af_heart.pt")
        voices = torch.load(voice_path, weights_only=True)
        # The voice pack is indexed by token count; pick the row matching this
        # sequence length (ref_s is a [1, 256] style vector).
        ref_s = voices[input_ids.shape[1] - 1]
        if dtype_override is not None:
            ref_s = ref_s.to(dtype_override)

        return {"input_ids": input_ids, "ref_s": ref_s}
