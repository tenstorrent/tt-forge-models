# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is a StyleTTS2-based TTS model: a PL-BERT text encoder feeds a prosody
predictor (with an LSTM duration head) and an iSTFTNet decoder that synthesizes
a waveform. The acoustic graph (``KModel.forward_with_tokens``) is exposed here
through a thin wrapper that takes pre-tokenized phoneme ids and a voice/style
reference tensor, so the model can be traced with plain tensor inputs.
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


# Phonemes for the fixed sample sentence "Hello world, this is a test."
# Produced once by misaki G2P; hardcoded so the loader does not depend on a
# runtime spaCy/espeak g2p pass (which downloads models and is non-deterministic
# for out-of-vocabulary words).
DEFAULT_PHONEMES = "həlˈO wˈɜɹld, ðɪs ɪz ɐ tˈɛst."

# Kokoro renders audio at 24 kHz.
SAMPLE_RATE = 24000


class _KokoroAcousticWrapper(torch.nn.Module):
    """Wraps ``KModel`` so its forward takes tensor inputs and returns the
    waveform tensor only (dropping the data-dependent ``pred_dur`` so the graph
    has a single tensor output).

    This re-implements ``KModel.forward_with_tokens`` inline, dropping the
    repeated ``.to(self.device)`` calls. ``KModel.device`` is a property that
    iterates ``self.parameters()`` via a generator expression, which
    ``torch.compile``/Dynamo cannot trace ("cannot access free variable
    'named_children'"). The tensors are already on the target device here, so
    the casts are unnecessary; removing them lets the graph trace through to the
    actual compute ops.
    """

    def __init__(self, kmodel, speed: float = 1.0):
        super().__init__()
        self.kmodel = kmodel
        self.speed = speed

    def forward(self, input_ids, ref_s):
        km = self.kmodel
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long,
        )
        text_mask = (
            torch.arange(input_lengths.max())
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        bert_dur = km.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = km.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = km.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = km.predictor.lstm(d)
        duration = km.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / self.speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        # Data-dependent output length: the alignment matrix width depends on the
        # predicted per-token durations (runtime values), not on a static shape.
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=input_ids.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=input_ids.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = km.predictor.F0Ntrain(en, s)
        t_en = km.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = km.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio


class ModelVariant(StrEnum):
    """Available Kokoro model variants."""

    KOKORO_82M = "82M"


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

    # Default voice pack used to build the style/reference tensor
    DEFAULT_VOICE = "af_heart"

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
        from kokoro import KModel

        repo_id = self._variant_config.pretrained_model_name
        kmodel = KModel(repo_id=repo_id).eval()
        # NOTE: Kokoro cannot run in bfloat16. Its prosody predictor keeps the
        # LSTM hidden state / several weight-normed convs in float32 internally,
        # so a blanket .to(bfloat16) yields a mixed-dtype graph that fails with
        # "mat1 and mat2 must have the same dtype, but got Float and BFloat16"
        # inside _VF.lstm. We therefore run float32 and ignore a bfloat16
        # override; any non-bf16 override is still applied.
        if dtype_override is not None and dtype_override != torch.bfloat16:
            kmodel = kmodel.to(dtype_override)
        self._kmodel = kmodel

        model = _KokoroAcousticWrapper(kmodel).eval()
        return model

    def _build_input_ids(self):
        if self._kmodel is None:
            self.load_model()
        vocab = self._kmodel.vocab
        ids = [vocab.get(p) for p in DEFAULT_PHONEMES]
        ids = [i for i in ids if i is not None]
        input_ids = torch.LongTensor([[0, *ids, 0]])
        self._input_ids = input_ids
        return input_ids

    def load_inputs(self, dtype_override=None):
        from huggingface_hub import hf_hub_download

        input_ids = self._build_input_ids()

        repo_id = self._variant_config.pretrained_model_name
        voice_path = hf_hub_download(repo_id=repo_id, filename=f"voices/{self.DEFAULT_VOICE}.pt")
        voice = torch.load(voice_path, weights_only=True)
        # Voice packs are indexed by phoneme length: ref_s = voice[len(phonemes) - 1].
        ref_s = voice[input_ids.shape[1] - 3]

        # Keep ref_s in float32 to match the model (see load_model note on bf16).
        if dtype_override is not None and dtype_override != torch.bfloat16:
            ref_s = ref_s.to(dtype_override)

        return {"input_ids": input_ids, "ref_s": ref_s}
