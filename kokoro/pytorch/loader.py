# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is an 82M-parameter StyleTTS2-style TTS model: a PL-BERT (ALBERT)
text encoder feeds a prosody/duration predictor (with an LSTM) and a
TextEncoder, whose outputs drive an iSTFTNet vocoder (Decoder) to produce
the waveform. The model is driven through the ``kokoro`` PyPI package.

The public ``KModel.forward`` takes a phoneme *string*; the device path
needs a forward over tensors, so this loader wraps ``KModel`` and exposes
``forward_with_tokens(input_ids, ref_s)`` as the module's ``forward``.
"""
import torch
from typing import Optional

from ...base import ForgeModel
from ...config import (
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


class _KokoroForwardModule(torch.nn.Module):
    """Tensor-in / tensor-out wrapper around ``kokoro.KModel``.

    ``KModel.forward`` accepts a phoneme string and does the vocab lookup
    internally. ``forward_with_tokens`` already operates on tensors
    (``input_ids``, ``ref_s``), so expose that as this module's forward so
    the model can be traced/compiled with tensor inputs.
    """

    def __init__(self, kmodel, speed: float = 1.0):
        super().__init__()
        self.kmodel = kmodel
        self.speed = speed

    def forward(self, input_ids, ref_s):
        audio, _pred_dur = self.kmodel.forward_with_tokens(input_ids, ref_s, self.speed)
        return audio


class ModelLoader(ForgeModel):
    """Kokoro-82M text-to-speech loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # A fixed, fully in-vocab phoneme string: the misaki G2P of
    # "Hello world, this is a test." Baked in so load_inputs is deterministic
    # and needs no espeak/spacy G2P frontend at inference time. To synthesize
    # arbitrary text, run the phonemes through `misaki` (see requirements.txt).
    DEFAULT_PHONEMES = "həlˈO wˈɜɹld, ðɪs ɪz ɐ tˈɛst."
    DEFAULT_VOICE = "af_heart"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the Kokoro loader.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
        """
        super().__init__(variant)
        self._kmodel = None
        self._phoneme_ids = None

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
        """Load and return the Kokoro model wrapped for tensor inputs.

        Args:
            dtype_override: Optional torch.dtype to override the model's
                default (fp32) weights.

        Returns:
            torch.nn.Module: The wrapped Kokoro model.
        """
        from kokoro import KModel

        repo_id = self._variant_config.pretrained_model_name
        kmodel = KModel(repo_id=repo_id).eval()
        self._kmodel = kmodel

        # Resolve the fixed phoneme string to ids against the model vocab.
        ids = [kmodel.vocab.get(p) for p in self.DEFAULT_PHONEMES]
        self._phoneme_ids = [i for i in ids if i is not None]

        model = _KokoroForwardModule(kmodel).eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs (phoneme ids + reference style vector).

        Args:
            dtype_override: Optional torch.dtype to override the default dtype
                of the floating-point reference style vector.

        Returns:
            dict: {"input_ids": LongTensor[1, T], "ref_s": FloatTensor[1, 256]}
        """
        from huggingface_hub import hf_hub_download

        if self._phoneme_ids is None:
            # Ensure the vocab is available to resolve phoneme ids.
            self.load_model(dtype_override=dtype_override)

        repo_id = self._variant_config.pretrained_model_name
        voice_path = hf_hub_download(
            repo_id=repo_id, filename=f"voices/{self.DEFAULT_VOICE}.pt"
        )
        # Voice pack is [context_len, 1, 256]; pick the row for this length.
        pack = torch.load(voice_path, weights_only=True)
        ref_s = pack[len(self._phoneme_ids) - 1]  # [1, 256]

        input_ids = torch.LongTensor([[0, *self._phoneme_ids, 0]])

        if dtype_override is not None:
            ref_s = ref_s.to(dtype_override)

        return {"input_ids": input_ids, "ref_s": ref_s}
