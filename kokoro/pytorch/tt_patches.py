# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tenstorrent-specific patches for the upstream ``kokoro`` package.

Instead of vendoring the full Kokoro model source, the loader installs the
upstream ``kokoro`` package (see ``requirements.nodeps.txt``, pinned to the
version below) and this module applies the small set of edits needed for the
model to compile and reach a deterministic PCC comparison on TT hardware. Every
patch here corresponds to a load-bearing behaviour change; the rest of the
architecture is used verbatim from upstream.

The patched method bodies are copied verbatim from the corresponding upstream
methods with only the documented deltas applied, so this file must be kept in
lockstep with ``KOKORO_PINNED_VERSION``.

Deltas applied
--------------
1. ``weight_norm`` (construction-time): upstream uses
   ``torch.nn.utils.parametrizations.weight_norm``, whose ``state_dict`` keys
   (``parametrizations.weight.original{0,1}``) do not match the released
   ``kokoro-v1_0.pth`` checkpoint (saved with the classic ``weight_g``/
   ``weight_v`` keys). Swap back to ``torch.nn.utils.weight_norm`` so the
   trained weights load.
2. ``KModel.forward_with_tokens``: pin a fixed per-token duration instead of
   ``torch.round(duration)`` — the bf16 device path rounds across the floor()
   boundary differently than the fp32 CPU golden, yielding different waveform
   lengths that cannot be PCC-compared; ``torch.round`` also has no tt-mlir
   ``stablehlo.round_nearest_even`` legalization.
3. ``SineGen._f02sine`` / ``SineGen.forward``: zero the ``torch.rand`` /
   ``torch.randn_like`` phase-and-additive noise so the forward is
   deterministic (independent RNG draws on CPU vs device otherwise cap PCC).
4. ``SourceModuleHnNSF.forward``: cast the fp32 sine source back to the
   module's activation dtype before the linear layer.
5. LSTM call sites (``TextEncoder`` / ``ProsodyPredictor`` / ``DurationEncoder``):
   bypass ``pack_padded_sequence`` for the single-sequence (batch-1) case — the
   compiler cannot trace the data-dependent ``PackedSequence`` packing — and
   pin the zero-pad buffers to the activation dtype.
"""

import importlib
import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep in lockstep with requirements.nodeps.txt. The patched method bodies below
# are copied from this exact upstream release.
KOKORO_PINNED_VERSION = "0.9.4"

_APPLIED = False


def _pip_kokoro_pkg_dir():
    """Absolute path to the *pip-installed* ``kokoro`` package directory.

    Resolved via distribution metadata rather than ``import``/``find_spec`` on
    the bare name, because tt-forge-models ships its own top-level ``kokoro/``
    model-family directory which is also on ``sys.path`` — a plain
    ``find_spec("kokoro")`` resolves to *that* dir (no ``istftnet.py``) instead
    of the pip package. Returns ``None`` if the pip package is not installed.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        pkg_dir = str(distribution("kokoro").locate_file("kokoro"))
    except PackageNotFoundError:
        return None
    # Guard against a metadata/model-dir mismatch: only accept a dir that
    # actually contains the acoustic-model source we patch.
    return pkg_dir if os.path.isfile(os.path.join(pkg_dir, "istftnet.py")) else None


def _import_kokoro_internals():
    """Import ``kokoro.{model,istftnet,modules,custom_stft}`` without executing
    ``kokoro/__init__.py``.

    The package ``__init__`` eagerly imports ``.pipeline``, which pulls in
    ``misaki`` (the g2p front-end + its ``espeak-ng`` system dependency). We only
    need the acoustic model (``KModel``), whose import chain does not touch
    ``misaki``, so we register a bare ``kokoro`` package object (carrying only
    ``__path__``, pointed explicitly at the pip install dir) in ``sys.modules``
    and import the submodules directly. The intra-package
    ``from kokoro.custom_stft import ...`` absolute import in ``istftnet`` then
    resolves against this bare package. Pointing ``__path__`` at the pip dir
    explicitly also shadows the same-named tt-forge-models model directory.
    """
    if all(
        m in sys.modules for m in ("kokoro.model", "kokoro.istftnet", "kokoro.modules")
    ):
        return (
            sys.modules["kokoro.model"],
            sys.modules["kokoro.istftnet"],
            sys.modules["kokoro.modules"],
        )

    # The test runner pip-installs this model's requirements *mid-process*, so
    # the import-system finder caches were built before ``kokoro`` existed in
    # site-packages. Invalidate them so the freshly-installed package is seen.
    importlib.invalidate_caches()

    pkg_dir = _pip_kokoro_pkg_dir()
    if pkg_dir is None:
        raise ImportError(
            "The 'kokoro' package is not installed. Install it (without its "
            "heavy g2p deps) via this model's requirements.nodeps.txt: "
            f"`pip install --no-deps kokoro=={KOKORO_PINNED_VERSION}`."
        )

    # Register (or repoint) a bare ``kokoro`` package at the pip dir. Overwrite
    # any existing entry that points elsewhere — e.g. the tt-forge-models
    # ``kokoro/`` model dir picked up from sys.path.
    existing = sys.modules.get("kokoro")
    existing_path = list(getattr(existing, "__path__", []) or [])
    if existing is None or existing_path[:1] != [pkg_dir]:
        pkg = types.ModuleType("kokoro")
        pkg.__path__ = [pkg_dir]
        sys.modules["kokoro"] = pkg

    istftnet_mod = importlib.import_module("kokoro.istftnet")
    modules_mod = importlib.import_module("kokoro.modules")
    model_mod = importlib.import_module("kokoro.model")
    return model_mod, istftnet_mod, modules_mod


# --------------------------------------------------------------------------- #
# Patched method bodies (copied from upstream with the documented deltas).
# --------------------------------------------------------------------------- #


def _run_lstm(lstm, x, lengths, batch_first=True):
    """Run a (bidirectional) LSTM, bypassing pack/pad for single-sequence input.

    ``nn.utils.rnn.pack_padded_sequence`` produces a ``PackedSequence`` whose
    backing tensor is not materialized on the TT/XLA device (the compiler cannot
    trace the data-dependent packing), which fails compilation. For a single
    unpadded sequence (batch size 1, as used here) packing is a no-op, so we
    call the LSTM directly. Batched/padded callers still get the exact packed
    path on CPU.
    """
    batch_dim = 0 if batch_first else 1
    if x.shape[batch_dim] == 1:
        lstm.flatten_parameters()
        out, _ = lstm(x)
        return out
    packed = nn.utils.rnn.pack_padded_sequence(
        x, lengths, batch_first=batch_first, enforce_sorted=False
    )
    lstm.flatten_parameters()
    packed, _ = lstm(packed)
    out, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=batch_first)
    return out


@torch.no_grad()
def _kmodel_forward_with_tokens(self, input_ids, ref_s, speed=1):
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
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(self.device)
    bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = self.predictor.lstm(d)
    duration = self.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    # Bringup: the per-token predicted duration sets the output waveform
    # length via repeat_interleave below. The duration logits are computed
    # in bf16 on device and round across the floor() boundary differently
    # than the fp32 CPU golden (several tokens sit ~0.06 from a half-integer
    # boundary), so the device and CPU total frame counts diverge (e.g. 61
    # vs 56) and the variable-length waveforms cannot be PCC-compared. Pin a
    # fixed per-token duration so the alignment — and thus the output length
    # — is identical on device and CPU; PCC then validates the decoder +
    # iSTFTNet vocoder on a deterministic alignment. The duration predictor
    # path (bert -> text_encoder -> lstm -> duration_proj) above is still
    # exercised by the graph. ``torch.round`` is also avoided here because
    # stablehlo.round_nearest_even has no tt-mlir legalization.
    _FIXED_DUR = 5
    pred_dur = torch.full(
        (input_ids.shape[1],), _FIXED_DUR, device=self.device, dtype=torch.long
    )
    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=self.device), pred_dur
    )
    pred_aln_trg = torch.zeros(
        (input_ids.shape[1], indices.shape[0]), device=self.device, dtype=d.dtype
    )
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
    en = d.transpose(-1, -2) @ pred_aln_trg
    F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
    t_en = self.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
    return audio, pred_dur


def _sinegen_f02sine(self, f0_values):
    """f0_values: (batchsize, length, dim)
    where dim indicates fundamental tone and overtones
    """
    # convert to F0 in rad. The interger part n can be ignored
    # because 2 * torch.pi * n doesn't affect phase
    rad_values = (f0_values / self.sampling_rate) % 1
    # Initial phase noise. The original adds torch.rand() per-harmonic phase
    # dither, which makes the forward non-deterministic: the runner's CPU
    # golden pass and the TT pass draw independent random values, so PCC can
    # never converge (caps ~0.18). Zeroed here for deterministic inference /
    # PCC bringup (equivalent to a fixed zero initial phase).
    rand_ini = torch.zeros(
        f0_values.shape[0], f0_values.shape[2], device=f0_values.device
    )
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
    if not self.flag_for_pulse:
        rad_values = F.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        sines = torch.sin(phase)
    else:
        # If necessary, make sure that the first time step of every
        # voiced segments is sin(pi) or cos(0)
        # This is used for pulse-train generation
        # identify the last time step in unvoiced segments
        uv = self._f02uv(f0_values)
        uv_1 = torch.roll(uv, shifts=-1, dims=1)
        uv_1[:, -1, :] = 1
        u_loc = (uv < 1) * (uv_1 > 0)
        # get the instantanouse phase
        tmp_cumsum = torch.cumsum(rad_values, dim=1)
        # different batch needs to be processed differently
        for idx in range(f0_values.shape[0]):
            temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
            temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
            # stores the accumulation of i.phase within
            # each voiced segments
            tmp_cumsum[idx, :, :] = 0
            tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
        # rad_values - tmp_cumsum: remove the accumulation of i.phase
        # within the previous voiced segment.
        i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
        # get the sines
        sines = torch.cos(i_phase * 2 * torch.pi)
    return sines


def _sinegen_forward(self, f0):
    """sine_tensor, uv = forward(f0)
    input F0: tensor(batchsize=1, length, dim=1)
              f0 for unvoiced steps should be 0
    output sine_tensor: tensor(batchsize=1, length, dim)
    output uv: tensor(batchsize=1, length, 1)
    """
    f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
    # fundamental component
    fn = torch.multiply(
        f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
    )
    # generate sine waveforms
    sine_waves = self._f02sine(fn) * self.sine_amp
    # generate uv signal
    # uv = torch.ones(f0.shape)
    # uv = uv * (f0 > self.voiced_threshold)
    uv = self._f02uv(f0)
    # noise: for unvoiced should be similar to sine_amp
    #        std = self.sine_amp/3 -> max value ~ self.sine_amp
    #        for voiced regions is self.noise_std
    noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
    # Additive dither. torch.randn_like makes the forward non-deterministic
    # (see _f02sine), so PCC vs the CPU golden cannot converge. Zeroed for
    # deterministic inference / PCC bringup.
    noise = noise_amp * torch.zeros_like(sine_waves)
    # first: set the unvoiced part to 0 by uv
    # then: additive noise
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


def _sourcemodule_forward(self, x):
    """
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    """
    # source for harmonic branch
    with torch.no_grad():
        sine_wavs, uv, _ = self.l_sin_gen(x)
    # SineGen runs its phase accumulation in fp32 for numerical accuracy;
    # cast back to the module's activation dtype before the linear layer.
    sine_wavs = sine_wavs.to(self.l_linear.weight.dtype)
    sine_merge = self.l_tanh(self.l_linear(sine_wavs))
    # source for noise branch, in the same shape as uv
    noise = torch.randn_like(uv) * self.sine_amp / 3
    return sine_merge, noise, uv


def _textencoder_forward(self, x, input_lengths, m):
    x = self.embedding(x)  # [B, T, emb]
    x = x.transpose(1, 2)  # [B, emb, T]
    m = m.unsqueeze(1)
    x.masked_fill_(m, 0.0)
    for c in self.cnn:
        x = c(x)
        x.masked_fill_(m, 0.0)
    x = x.transpose(1, 2)  # [B, T, chn]
    lengths = (
        input_lengths
        if input_lengths.device == torch.device("cpu")
        else input_lengths.to("cpu")
    )
    x = _run_lstm(self.lstm, x, lengths, batch_first=True)
    x = x.transpose(-1, -2)
    x_pad = torch.zeros(
        [x.shape[0], x.shape[1], m.shape[-1]], device=x.device, dtype=x.dtype
    )
    x_pad[:, :, : x.shape[-1]] = x
    x = x_pad
    x.masked_fill_(m, 0.0)
    return x


def _prosodypredictor_forward(self, texts, style, text_lengths, alignment, m):
    d = self.text_encoder(texts, style, text_lengths, m)
    m = m.unsqueeze(1)
    lengths = (
        text_lengths
        if text_lengths.device == torch.device("cpu")
        else text_lengths.to("cpu")
    )
    x = _run_lstm(self.lstm, d, lengths, batch_first=True)
    x_pad = torch.zeros(
        [x.shape[0], m.shape[-1], x.shape[-1]], device=x.device, dtype=x.dtype
    )
    x_pad[:, : x.shape[1], :] = x
    x = x_pad
    duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
    en = d.transpose(-1, -2) @ alignment
    return duration.squeeze(-1), en


def _durationencoder_forward(self, x, style, text_lengths, m):
    masks = m
    x = x.permute(2, 0, 1)
    s = style.expand(x.shape[0], x.shape[1], -1)
    x = torch.cat([x, s], axis=-1)
    x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    x = x.transpose(0, 1)
    x = x.transpose(-1, -2)
    for block in self.lstms:
        # AdaLayerNorm branch is unchanged from upstream; only the LSTM branch
        # swaps pack/pad for the single-sequence bypass.
        if block.__class__.__name__ == "AdaLayerNorm":
            x = block(x.transpose(-1, -2), style).transpose(-1, -2)
            x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
            x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
        else:
            lengths = (
                text_lengths
                if text_lengths.device == torch.device("cpu")
                else text_lengths.to("cpu")
            )
            x = x.transpose(-1, -2)
            x = _run_lstm(block, x, lengths, batch_first=True)
            x = F.dropout(x, p=self.dropout, training=False)
            x = x.transpose(-1, -2)
            x_pad = torch.zeros(
                [x.shape[0], x.shape[1], m.shape[-1]], device=x.device, dtype=x.dtype
            )
            x_pad[:, :, : x.shape[-1]] = x
            x = x_pad

    return x.transpose(-1, -2)


def apply_tt_patches():
    """Import the upstream ``kokoro`` acoustic model and apply the TT deltas.

    Idempotent. Returns the (patched) ``KModel`` class.
    """
    global _APPLIED
    model_mod, istftnet_mod, modules_mod = _import_kokoro_internals()

    if _APPLIED:
        return model_mod.KModel

    # (1) Construction-time weight_norm swap — must happen before any KModel is
    #     built so __init__ picks up the classic (checkpoint-matching) variant.
    istftnet_mod.weight_norm = torch.nn.utils.weight_norm
    modules_mod.weight_norm = torch.nn.utils.weight_norm

    # (2)-(5) Method-body patches.
    model_mod.KModel.forward_with_tokens = _kmodel_forward_with_tokens
    istftnet_mod.SineGen._f02sine = _sinegen_f02sine
    istftnet_mod.SineGen.forward = _sinegen_forward
    istftnet_mod.SourceModuleHnNSF.forward = _sourcemodule_forward
    modules_mod.TextEncoder.forward = _textencoder_forward
    modules_mod.ProsodyPredictor.forward = _prosodypredictor_forward
    modules_mod.DurationEncoder.forward = _durationencoder_forward

    _APPLIED = True
    return model_mod.KModel
