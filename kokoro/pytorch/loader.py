# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is a StyleTTS2-style text-to-speech model: a PL-BERT phoneme encoder,
a prosody/duration predictor (LSTM-based), a text encoder, and an iSTFTNet
vocoder decoder. The public ``KModel.forward`` takes a *phoneme string*; this
loader wraps ``KModel.forward_with_tokens`` so the model can be driven with
plain tensors (``input_ids`` + ``ref_s`` style/voice vector), matching the
ForgeModel ``model(**inputs)`` contract used by the test harness.
"""
from typing import Optional

import torch

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

    KOKORO_82M = "82M"


class _KokoroTokenForward(torch.nn.Module):
    """Tensor-in / tensor-out wrapper around ``KModel.forward_with_tokens``.

    The underlying KModel exposes a tensor-level forward (``forward_with_tokens``)
    that maps ``input_ids`` and a voice style vector ``ref_s`` to an audio
    waveform. This wrapper exposes that as a standard ``forward(input_ids, ref_s)``
    so the runner can call ``model(**inputs)`` and compare the audio output.
    """

    def __init__(self, kmodel, speed: float = 1.0):
        super().__init__()
        self.kmodel = kmodel
        self.speed = speed

    def forward(self, input_ids, ref_s):
        # Reimplements KModel.forward_with_tokens, but routes every device move
        # through ``input_ids.device`` instead of the original ``self.device``
        # property. ``self.device`` does ``next(p.device for p in
        # self.parameters())``, a generator over parameters that TorchDynamo
        # cannot trace (NameError on a free variable), so it must be avoided in
        # the compiled forward. The harness has already placed the module and
        # inputs on the target device, so the explicit ``.to(...)`` calls are
        # redundant and dropped.
        m = self.kmodel
        device = input_ids.device
        speed = self.speed

        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=device,
            dtype=torch.long,
        )
        text_mask = (
            torch.arange(input_lengths.max())
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)

        bert_dur = m.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = m.predictor.lstm(d)
        duration = m.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        # Data-dependent alignment expansion: the audio length is the sum of the
        # predicted per-token durations, so ``indices`` / ``pred_aln_trg`` have a
        # runtime-dependent shape.
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = m.predictor.F0Ntrain(en, s)
        t_en = m.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = m.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio


class ModelLoader(ForgeModel):
    """Kokoro-82M model loader implementation for text-to-speech tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.KOKORO_82M: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.KOKORO_82M

    # A short phoneme (IPA) string for "Hello world", mapped to ids via the
    # model vocab. Baking phonemes in avoids a runtime espeak-ng / G2P dependency.
    DEFAULT_PHONEMES = "həlˈoʊ wˈɜːld"
    # Default voice pack used to source the reference style vector ``ref_s``.
    DEFAULT_VOICE = "af_heart"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

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
        """Load the Kokoro KModel wrapped for tensor-level inference.

        Note: ``dtype_override`` is intentionally not applied to the weights.
        Kokoro mixes float32-hardcoded internals (PL-BERT attention) with a
        complex-valued iSTFT vocoder, so casting the whole module to bfloat16
        breaks even on CPU (mat1/mat2 dtype mismatch). The model is kept in its
        native float32.
        """
        from kokoro import KModel

        repo_id = self._variant_config.pretrained_model_name
        kmodel = KModel(repo_id=repo_id).eval()
        self._model = _KokoroTokenForward(kmodel)
        return self._model

    def load_inputs(self, dtype_override=None):
        """Build tensor inputs (input_ids, ref_s) for the wrapped model.

        ``ref_s`` is kept in float32 to match the model (see ``load_model``).
        """
        from huggingface_hub import hf_hub_download

        if self._model is None:
            self.load_model(dtype_override=dtype_override)
        kmodel = self._model.kmodel

        # Map phonemes -> input_ids the same way KModel.forward does, then wrap
        # with the [0, ..., 0] start/end sentinels.
        ids = [
            i
            for i in (kmodel.vocab.get(p) for p in self.DEFAULT_PHONEMES)
            if i is not None
        ]
        input_ids = torch.LongTensor([[0, *ids, 0]])

        # Voice packs are indexed by the inner phoneme length; ref_s is [1, 256].
        repo_id = self._variant_config.pretrained_model_name
        voice_path = hf_hub_download(repo_id, f"voices/{self.DEFAULT_VOICE}.pt")
        pack = torch.load(voice_path, weights_only=True)
        ref_s = pack[len(ids)]

        return {"input_ids": input_ids, "ref_s": ref_s}
