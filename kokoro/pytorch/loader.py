# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is a StyleTTS2 / ISTFTNet based TTS model distributed as the ``kokoro``
package. It is a custom (non-``transformers``) architecture made up of a
phoneme-level ALBERT text encoder, a prosody predictor (duration + F0/N), a
convolutional text encoder and an ISTFTNet vocoder decoder.

The published ``KModel.forward`` takes a Python phoneme *string* and performs
CPU-side tokenization, which is not traceable/compilable. This loader instead
wraps the tensor entry point ``KModel.forward_with_tokens(input_ids, ref_s,
speed)`` in an ``nn.Module`` whose ``forward(input_ids, ref_s)`` returns the
synthesized audio waveform, so the model can be exercised as a single tensor-in
/ tensor-out forward pass on device.
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

    BASE = "82m"


class _KokoroWrapper(torch.nn.Module):
    """Expose ``KModel``'s tensor entry point as a standard forward pass.

    The runner feeds inputs as ``model(input_ids=..., ref_s=...)``. ``KModel``'s
    own ``forward`` expects a phoneme string, so this wrapper re-implements the
    tensor path ``KModel.forward_with_tokens`` and returns only the audio
    waveform (dropping the predicted-duration tensor).

    The re-implementation derives every device from the input tensors instead
    of ``KModel.device``. ``KModel.device`` is a property that does
    ``next(p.device for p in self.parameters())``; under ``torch.compile`` /
    dynamo that generator raises ``NameError: cannot access free variable
    'named_children'``, so the original ``forward_with_tokens`` cannot be
    traced. Inputs are already placed on the target device by the test harness,
    so the redundant ``.to(self.device)`` calls are simply dropped here.
    """

    def __init__(self, kmodel: torch.nn.Module):
        super().__init__()
        self.kmodel = kmodel

    def forward(self, input_ids, ref_s):
        km = self.kmodel
        device = input_ids.device

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
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        bert_dur = km.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = km.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = km.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = km.predictor.lstm(d)
        duration = km.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        # Data-dependent alignment upsampling: the frame count (and therefore the
        # output audio length) depends on the predicted per-token durations.
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = km.predictor.F0Ntrain(en, s)
        t_en = km.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = km.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio


class ModelLoader(ForgeModel):
    """Kokoro-82M model loader implementation for text-to-speech tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # A short IPA phoneme string ("Hello world.") in Kokoro's phoneme vocab.
    # Characters not present in the model vocab are dropped during tokenization.
    sample_phonemes = "həlˈoʊ wˈɜːld."

    # Voice style pack used to source the reference style vector ``ref_s``.
    sample_voice = "voices/af_heart.pt"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="kokoro",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self):
        """Load and return the Kokoro model instance for this instance's variant.

        Note:
            No ``dtype_override`` is exposed. Kokoro is a float32-native model
            whose ISTFTNet decoder and LSTM-based prosody predictor mix in
            hard-coded float32 constants; casting the weights to bfloat16 raises
            a dtype mismatch (``mat1 Float vs mat2 BFloat16``) inside the LSTM.
            The model is therefore brought up in its native float32.

        Returns:
            torch.nn.Module: A wrapper around ``KModel`` whose forward takes
            ``(input_ids, ref_s)`` tensors and returns the audio waveform.
        """
        from kokoro import KModel

        repo_id = self._variant_config.pretrained_model_name

        # disable_complex=True makes the ISTFTNet decoder avoid torch.stft/istft
        # complex-number ops, which are not supported on TT hardware.
        kmodel = KModel(repo_id=repo_id, disable_complex=True).eval()

        model = _KokoroWrapper(kmodel)
        model.eval()
        self.model = model
        return model

    def load_inputs(self):
        """Load and return sample inputs for the Kokoro model.

        Returns:
            dict: ``{"input_ids": LongTensor[1, N], "ref_s": FloatTensor[1, 256]}``
        """
        from huggingface_hub import hf_hub_download

        if self.model is None:
            self.load_model()

        vocab = self.model.kmodel.vocab
        ids = [vocab.get(p) for p in self.sample_phonemes]
        ids = [i for i in ids if i is not None]

        # forward_with_tokens expects the boundary tokens (0) that KModel.forward
        # normally prepends/appends around the phoneme ids.
        input_ids = torch.LongTensor([[0, *ids, 0]])

        # Reference style vector: voice packs are indexed by phoneme length.
        pack_path = hf_hub_download(
            self._variant_config.pretrained_model_name, self.sample_voice
        )
        pack = torch.load(pack_path, weights_only=True)
        ref_s = pack[len(ids) - 1].clone()  # [1, 256]

        return {"input_ids": input_ids, "ref_s": ref_s}
