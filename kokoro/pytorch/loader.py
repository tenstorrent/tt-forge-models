# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is a StyleTTS2-based TTS model: a PL-BERT text encoder feeds a prosody
predictor (duration + F0/energy) and a TextEncoder, whose outputs drive an
iSTFTNet decoder that synthesizes the waveform. The model consumes phoneme
token ids (vocab of 178) plus a per-speaker style vector (``ref_s``), not raw
text, so we drive it through ``KModel.forward_with_tokens`` which is a pure
tensor function suitable for tracing/compilation.
"""
import types
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


def _unrolled_lstm_forward(self, input, hx=None):
    """Op-level, time-unrolled replacement for ``nn.LSTM.forward``.

    The native fused ``_VF.lstm`` kernel cannot be traced/compiled by the TT
    backend (it surfaces as an opaque custom kernel). This reimplements a
    single-layer (optionally bidirectional) LSTM using only basic ops
    (matmul / sigmoid / tanh / elementwise), which the compiler can lower.
    Numerically matches ``nn.LSTM`` to ~1e-6. Assumes ``batch_first=True`` and
    ``num_layers == 1`` (all Kokoro LSTMs satisfy this).
    """
    B, L, _ = input.shape
    H = self.hidden_size

    def run_dir(w_ih, w_hh, b_ih, b_hh, reverse):
        h = input.new_zeros(B, H)
        c = input.new_zeros(B, H)
        steps = range(L - 1, -1, -1) if reverse else range(L)
        collected = []
        for t in steps:
            xt = input[:, t, :]
            gates = xt @ w_ih.t() + b_ih + h @ w_hh.t() + b_hh
            i, f, g, o = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            collected.append(h)
        if reverse:
            collected = collected[::-1]
        return torch.stack(collected, dim=1)  # [B, L, H]

    fwd = run_dir(
        self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0, False
    )
    if self.bidirectional:
        bwd = run_dir(
            self.weight_ih_l0_reverse,
            self.weight_hh_l0_reverse,
            self.bias_ih_l0_reverse,
            self.bias_hh_l0_reverse,
            True,
        )
        out = torch.cat([fwd, bwd], dim=2)  # [B, L, 2H]
    else:
        out = fwd
    return out, None


def _text_encoder_forward(self, x, input_lengths, m):
    """TextEncoder.forward without pack/pad.

    The original packs the (here single, full-length, unpadded) sequence before
    the LSTM and forces ``input_lengths`` onto the CPU — both uncompilable. With
    no padding, pack→lstm→pad is identical to running the LSTM on the full
    sequence, so we drop it.
    """
    x = self.embedding(x)
    x = x.transpose(1, 2)
    m = m.unsqueeze(1)
    x = x.masked_fill(m, 0.0)
    for c in self.cnn:
        x = c(x)
        x = x.masked_fill(m, 0.0)
    x = x.transpose(1, 2)
    x, _ = self.lstm(x)
    x = x.transpose(-1, -2)
    x = x.masked_fill(m, 0.0)
    return x


def _duration_encoder_forward(self, x, style, text_lengths, m):
    """DurationEncoder.forward without pack/pad (see _text_encoder_forward)."""
    import torch.nn.functional as F
    from kokoro.modules import AdaLayerNorm

    masks = m
    x = x.permute(2, 0, 1)
    s = style.expand(x.shape[0], x.shape[1], -1)
    x = torch.cat([x, s], axis=-1)
    x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    x = x.transpose(0, 1)
    x = x.transpose(-1, -2)
    for block in self.lstms:
        if isinstance(block, AdaLayerNorm):
            x = block(x.transpose(-1, -2), style).transpose(-1, -2)
            x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
            x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
        else:
            x = x.transpose(-1, -2)
            x, _ = block(x)
            x = F.dropout(x, p=self.dropout, training=False)
            x = x.transpose(-1, -2)
    return x.transpose(-1, -2)


def _patch_for_compile(model):
    """Make Kokoro traceable on the device.

    1. Replace every fused ``nn.LSTM`` forward with the unrolled implementation.
    2. Replace the ``TextEncoder`` / ``DurationEncoder`` forwards with pack-free
       versions that don't call ``pack_padded_sequence`` (which forces a host
       transfer and a data-dependent packed layout).
    """
    from kokoro.modules import TextEncoder, DurationEncoder

    patched_lstms = 0
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            module.forward = types.MethodType(_unrolled_lstm_forward, module)
            patched_lstms += 1
        elif isinstance(module, TextEncoder):
            module.forward = types.MethodType(_text_encoder_forward, module)
        elif isinstance(module, DurationEncoder):
            module.forward = types.MethodType(_duration_encoder_forward, module)
    return patched_lstms


def _text_mask(input_ids):
    """Build the (all-False, full-length) phoneme mask Kokoro expects."""
    dev = input_ids.device
    input_lengths = torch.full(
        (input_ids.shape[0],), input_ids.shape[-1], device=dev, dtype=torch.long
    )
    text_mask = (
        torch.arange(input_lengths.max())
        .unsqueeze(0)
        .expand(input_lengths.shape[0], -1)
        .type_as(input_lengths)
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(dev)
    return input_lengths, text_mask


def predict_alignment(kmodel, input_ids, ref_s, max_frames, speed=1.0):
    """Run Kokoro's duration predictor and return a fixed alignment matrix.

    This is the *discrete*, non-differentiable part of the pipeline: the BERT /
    prosody predictor emits a per-phoneme duration which is rounded to an integer
    number of acoustic frames, and the encoder outputs are upsampled by repeating
    each phoneme that many times. A single off-by-one in those integer durations
    time-shifts the entire downstream waveform, so reproducing it bit-for-bit on
    the device is both compiler-hostile (data-dependent shapes) and accuracy-
    fragile. We therefore compute it once on the host (analogous to keeping a
    diffusion scheduler in host Python) and feed the resulting
    ``[1, seq, max_frames]`` alignment to the device model as a fixed input, so
    the device path and the CPU golden share an identical, shift-free alignment.

    ``max_frames`` must equal the predicted total duration exactly — the iSTFTNet
    source generator accumulates phase over the whole frame axis, so trailing
    silent frames corrupt the entire waveform.
    """
    input_lengths, text_mask = _text_mask(input_ids)
    bert_dur = kmodel.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = kmodel.predictor.lstm(d)
    duration = kmodel.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze(0)  # [seq]
    total = int(pred_dur.sum())
    if total != max_frames:
        raise ValueError(
            f"predicted total duration {total} != max_frames {max_frames}; "
            "update ModelLoader.MAX_FRAMES to match the deterministic sample."
        )
    # Monotonic one-hot upsampling matrix [seq, max_frames].
    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=input_ids.device), pred_dur
    )
    pred_aln_trg = torch.zeros((input_ids.shape[1], max_frames), device=input_ids.device)
    pred_aln_trg[indices, torch.arange(max_frames, device=input_ids.device)] = 1
    return pred_aln_trg.unsqueeze(0)  # [1, seq, max_frames]


class _KokoroWrapper(torch.nn.Module):
    """Tensor-only forward over ``KModel`` for the compiled (device) path.

    The runner invokes ``model(**inputs)`` and compares a single output tensor,
    so this returns just the synthesized waveform. The forward is reimplemented
    (rather than calling ``KModel.forward_with_tokens``) for three reasons:

    1. The device is captured from the input instead of via ``KModel.device``,
       whose generator-expression property the dynamo tracer cannot evaluate.
    2. The fused LSTM kernels and ``pack_padded_sequence`` calls are replaced by
       traceable equivalents (see ``_patch_for_compile``).
    3. The discrete duration→alignment upsampling is precomputed on the host and
       supplied as the ``pred_aln_trg`` input (see ``predict_alignment``), so the
       device graph is static and the alignment is identical to the CPU golden.

    Everything else — BERT, the duration encoder, F0/energy prediction and the
    iSTFTNet decoder — runs on the device.
    """

    def __init__(self, kmodel):
        super().__init__()
        self.kmodel = kmodel

    def forward(self, input_ids, ref_s, pred_aln_trg):
        m = self.kmodel
        input_lengths, text_mask = _text_mask(input_ids)
        bert_dur = m.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask)
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
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Voice (speaker style) pack used to source the reference style vector.
    DEFAULT_VOICE = "af_heart"

    # Phoneme sequence length for the sample inputs. Durations (and thus the
    # synthesized audio length) are deterministic for a given input, so the
    # CPU golden and device output share the same shape for comparison.
    DEFAULT_SEQ_LEN = 32

    # Total acoustic frames the alignment upsamples to. For the deterministic
    # sample (DEFAULT_SEQ_LEN phonemes, fixed seed, af_heart voice) the duration
    # predictor sums to exactly 97 frames; predict_alignment asserts this so a
    # mismatch surfaces loudly rather than silently corrupting the output.
    MAX_FRAMES = 97

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None
        self._kmodel = None

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

    def load_model(self, dtype_override=None):
        """Load and return the Kokoro model instance for this instance's variant.

        Args:
            dtype_override: Accepted for interface compatibility but intentionally
                not applied. ``KModel.forward_with_tokens`` constructs float32
                intermediate tensors (e.g. the alignment matrix and masks) inline,
                so casting the weights to a lower-precision dtype produces
                mixed-dtype matmuls (Float vs BFloat16) inside the LSTM/decoder.
                The model therefore runs end-to-end in float32 for a consistent
                dtype across all ops.

        Returns:
            torch.nn.Module: The Kokoro model wrapped for tensor-only forward.
        """
        from kokoro import KModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        # disable_complex=True swaps the iSTFT for a real-valued implementation,
        # avoiding complex-tensor ops that are unsupported on the device.
        kmodel = KModel(repo_id=pretrained_model_name, disable_complex=True).eval()

        # Make the model traceable on device: unroll fused LSTM kernels and
        # drop pack_padded_sequence from the encoders (see _patch_for_compile).
        _patch_for_compile(kmodel)

        model = _KokoroWrapper(kmodel)
        model.eval()
        self.model = model
        self._kmodel = kmodel
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Kokoro model.

        Args:
            dtype_override: Accepted for interface compatibility but intentionally
                not applied — the model runs in float32 (see ``load_model``), so
                the float style vector stays float32 to match the weights.

        Returns:
            dict: ``input_ids`` (phonemes), ``ref_s`` (style) and the host-
            precomputed ``pred_aln_trg`` alignment matrix.
        """
        from huggingface_hub import hf_hub_download

        pretrained_model_name = self._variant_config.pretrained_model_name
        seq_len = self.DEFAULT_SEQ_LEN

        # Deterministic phoneme token ids in the valid vocab range [1, 177].
        generator = torch.Generator().manual_seed(0)
        input_ids = torch.randint(
            1, 178, (1, seq_len), dtype=torch.long, generator=generator
        )

        # Per-speaker style vector: voice packs are indexed by phoneme length.
        voice_path = hf_hub_download(
            pretrained_model_name, f"voices/{self.DEFAULT_VOICE}.pt"
        )
        voice = torch.load(voice_path, weights_only=True)
        ref_s = voice[seq_len]  # shape [1, 256], float32

        # Precompute the discrete duration→frame alignment on the host so the
        # device path and CPU golden share an identical (shift-free) alignment.
        kmodel = self._kmodel
        if kmodel is None:
            self.load_model()
            kmodel = self._kmodel
        with torch.no_grad():
            pred_aln_trg = predict_alignment(
                kmodel, input_ids, ref_s, self.MAX_FRAMES
            )

        return {
            "input_ids": input_ids,
            "ref_s": ref_s,
            "pred_aln_trg": pred_aln_trg,
        }
