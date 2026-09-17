# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 DiT transformer loader for text-to-video (+audio) generation.

Unlike the diffusers-based ``ltx2`` family loader, this loader drives the
*native* ``ltx_core`` source, which is consumed as a pinned git submodule at
``third_party/LTX-2`` (upstream ships LTX-2 and LTX-2.3 from one repo; they
differ by checkpoint, not by repository). The 22B LTX-2.3 audio-video DiT is
built straight from the checkpoint's embedded transformer config via
``LTXModelConfigurator.from_config`` with random weights -- no checkpoint is
downloaded and no HF pipeline is instantiated.

The submodule files are used **unmodified**. The four TT bring-up changes are
applied here as runtime patches (see ``_apply_runtime_patches``), each guarded
against upstream drift, and the missing ``torchaudio`` dependency is covered by
a loud ``sys.modules`` stub.

Repository: https://github.com/Lightricks/LTX-2
Weights:    https://huggingface.co/Lightricks/LTX-2.3

The native ``LTXModel.forward`` takes structured ``Modality`` objects rather
than plain tensors:

    forward(video: Modality | None, audio: Modality | None,
            perturbations: BatchedPerturbationConfig) -> (video_out, audio_out)

so ``load_model`` returns an ``nn.Module`` wrapper whose ``forward(*tensors)``
rebuilds the ``Modality`` objects and the (no-op) perturbation config, calls
the underlying model, and returns the video output tensor. ``load_inputs``
returns the matching plain tensors in the wrapper's forward-arg order.

Both variants build the SAME architecture from the SAME embedded config; they
differ only in the checkpoint they would load (which this scaffold does NOT do):

    Fast -> ltx-2.3-22b-distilled-1.1.safetensors
    Pro  -> ltx-2.3-22b-dev.safetensors

NOTE: the full 48-layer model is ~21B params -- host-CPU instantiation is
infeasible. Treat the transformer as derived / not-CPU-instantiated, exactly
like the diffusers ``ltx2`` reference. The reduced-layer CPU forward used to
validate the plumbing overrides ``num_layers`` to a small value.
"""

import importlib
import inspect
import os
import sys
import textwrap
import types
from typing import Optional

import torch

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

# ── Upstream source: pinned git submodule ────────────────────────────────────
_SUBMODULE_HINT = (
    "The LTX-2 submodule is not checked out. Run:\n"
    "  git submodule update --init third_party/LTX-2"
)


def _ltx_core_pkg_dir():
    """Absolute path to the upstream ``ltx_core`` package in the submodule.

    Resolved from this file's location: the loader lives at
    ``<repo>/ltx2_3/pytorch/loader.py`` and the submodule at
    ``<repo>/third_party/LTX-2``. Returns ``None`` if the submodule has not
    been checked out.
    """
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    pkg = os.path.join(
        repo_root, "third_party", "LTX-2", "packages", "ltx-core", "src", "ltx_core"
    )
    return pkg if os.path.isdir(pkg) else None


def _register_bare_package(name, path):
    """Register a bare package at ``path`` in ``sys.modules`` (no ``__init__``).

    ``ltx_core/__init__.py`` is empty upstream, so skipping it costs nothing;
    setting ``__path__`` explicitly lets every sub-package resolve normally with
    its own real ``__init__.py`` intact, without mutating ``sys.path`` (which
    would otherwise leak the submodule's ``src/`` dir into every import).
    """
    existing = sys.modules.get(name)
    if existing is None or list(getattr(existing, "__path__", []) or [])[:1] != [path]:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod


class _TorchaudioStub(types.ModuleType):
    """Stand-in for ``torchaudio``, which is not installed in the TT venv.

    ``ltx_core/model/audio_vae/ops.py`` imports ``torchaudio`` at module level
    for ``AudioProcessor`` (waveform -> mel), which is *not* on the audio-VAE
    encode/decode or vocoder paths this bring-up traces. Rather than patch the
    import out of upstream, register a stub that satisfies the import and
    raises loudly the moment anything actually touches it.
    """

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            # importlib and dynamo's skipfiles check probe module dunders
            # (``__file__`` above all) on everything in ``sys.modules``. Those
            # probes must miss quietly, like a namespace package -- raising here
            # aborts tracing of every model that merely has the stub loaded.
            raise AttributeError(attr)
        raise RuntimeError(
            f"torchaudio.{attr} was accessed, but torchaudio is not installed in "
            "this environment. The LTX-2.3 loader stubs it because the traced "
            "audio-VAE / vocoder paths never use it; a real dependency on "
            "torchaudio needs it installed."
        )


# ── Runtime patches over the pinned upstream source ──────────────────────────
def _indent_block(text, indent):
    """Re-indent a patch block authored at zero baseline to ``indent``.

    Patch texts are written with their first line flush and continuation lines
    relative to it, so one literal matches the statement at whatever depth
    ``inspect.getsource`` + ``textwrap.dedent`` leave it.
    """
    head, _, tail = text.partition("\n")
    return head if not tail else head + "\n" + textwrap.indent(tail, indent)


def _statement_indent(src, old_text, flag):
    """Leading whitespace of the line ``old_text`` starts on, in ``src``."""
    head = old_text.partition("\n")[0]
    indents = {
        line[: len(line) - len(line.lstrip())]
        for line in src.splitlines()
        if line.strip() and line.lstrip().startswith(head)
    }
    if len(indents) != 1:
        raise RuntimeError(
            f"patch drift ({flag}): expected the patched statement to start on "
            f"exactly one indentation level, found {sorted(indents)} -- the "
            "pinned ltx_core source has changed; re-derive this patch."
        )
    return indents.pop()


def _patch_method_source(cls, method_name, old_text, new_text, flag):
    """Recompile one upstream method with an exact-text substitution.

    Upstream method bodies are deliberately NOT copied into this file: the body
    is read back from the pinned submodule with ``inspect.getsource``, the exact
    ``old_text`` is substituted, and the result is compiled against the defining
    module's globals so free names still resolve upstream. That keeps the patch
    a reviewable one-hunk diff *and* gives it a drift guard for free -- if
    ``old_text`` is no longer present, the pin has moved and the patch must be
    re-derived rather than silently becoming a no-op.
    """
    original = cls.__dict__.get(method_name)
    if original is None:
        raise RuntimeError(
            f"patch drift ({flag}): {cls.__name__}.{method_name} no longer exists "
            "in the pinned ltx_core source -- re-derive this patch."
        )
    if getattr(original, flag, False):
        return

    src = textwrap.dedent(inspect.getsource(original))
    indent = _statement_indent(src, old_text, flag)
    old_block = _indent_block(old_text, indent)
    new_block = _indent_block(new_text, indent)
    if src.count(old_block) != 1:
        raise RuntimeError(
            f"patch drift ({flag}): expected exactly one occurrence of the patched "
            f"text in {cls.__name__}.{method_name}, found {src.count(old_block)} -- "
            "the pinned ltx_core source has changed; re-derive this patch."
        )

    module = sys.modules[cls.__module__]
    namespace = {}
    exec(  # noqa: S102 - recompiling upstream source with one audited edit
        compile(
            src.replace(old_block, new_block),
            f"<tt-forge patch {flag} of {cls.__module__}.{cls.__qualname__}>",
            "exec",
        ),
        module.__dict__,
        namespace,
    )
    patched = namespace[method_name]
    patched.__qualname__ = original.__qualname__
    setattr(patched, flag, True)
    setattr(cls, method_name, patched)


# Patch 1 -- STFT-as-conv reformulation (vocoder).
# TTNN lowers the strided ``F.conv1d`` over the full ~21k-sample width into a
# ``ttnn.conv2d`` whose L1 footprint overflows ("Not enough L1 memory", in fp32
# and bf16 alike, no fallback config helps). The effective compute is one small
# ``(n_frames x win_length) @ (win_length x n_freqs*2)`` matmul, so frame
# explicitly with ``F.unfold`` and matmul the fixed basis. ``conv1d`` is
# cross-correlation (no kernel flip) and unfold preserves window order, so this
# is numerically identical to the original conv.
_STFT_OLD = "spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)"
_STFT_NEW = """frames = F.unfold(
    y.unsqueeze(2),  # (B, 1, 1, T_pad)
    kernel_size=(1, self.win_length),
    stride=(1, self.hop_length),
)  # (B, win_length, n_frames)
basis = self.forward_basis.reshape(self.forward_basis.shape[0], self.win_length)
spec = torch.matmul(basis.to(frames.dtype), frames)"""

# Patch 2 -- honour ``_forge_compute_dtype`` in ``VocoderWithBWE.forward``.
# Upstream runs the whole pass in fp32 (bf16 accumulation over 108 sequential
# convs degrades spectral metrics 40-90%). Two problems on TT: CPU
# autocast-to-fp32 is a no-op, so fp32 activations meet bf16 bias ("Input type
# (float) and bias type (BFloat16) should be the same"), and the fp32 activation
# footprint overflows device L1 on the conv stack. When the loader stamps a
# compute dtype, disable autocast and cast the input to match the weights.
_VOCODER_OLD = """with torch.autocast(device_type=mel_spec.device.type, dtype=torch.float32):
    x = self.vocoder(mel_spec.float())"""
_VOCODER_NEW = """compute_dtype = getattr(self, "_forge_compute_dtype", torch.float32)
with torch.autocast(
    device_type=mel_spec.device.type,
    dtype=compute_dtype,
    enabled=compute_dtype == torch.float32,
):
    x = self.vocoder(mel_spec.to(compute_dtype))"""

# Patch 3 -- force the traceable RoPE frequency-grid generator.
# ``double_precision_rope`` (set by ``frequencies_precision: float64``) selects
# ``generate_freq_grid_np``, which runs float64 numpy inside the traced forward
# and breaks dynamo ("'ndarray' object has no attribute 'div'"). The grid depends
# only on scalar hyper-parameters and TT executes in bf16, so the float64->float32
# difference is washed out; forcing the torch-native generator keeps the forward
# a single graph.
_ROPE_OLD = (
    "freq_grid_generator = generate_freq_grid_np if self.double_precision_rope "
    "else generate_freq_grid_pytorch"
)
_ROPE_NEW = "freq_grid_generator = generate_freq_grid_pytorch"

# Patch 4 -- honour ``_forge_weights_dtype`` in ``VideoDecoder.forward``.
# ``next(self.parameters())`` is not traceable by dynamo (free-variable
# NameError). Read the stamped constant when present, else fall back to the
# original eager lookup so standalone use is unchanged.
_DECODER_DTYPE_OLD = "weights_dtype = next(self.parameters()).dtype"
_DECODER_DTYPE_NEW = """weights_dtype = getattr(self, "_forge_weights_dtype", None)
if weights_dtype is None:
    weights_dtype = next(self.parameters()).dtype"""


def _apply_runtime_patches():
    """Apply the four TT bring-up patches to the pinned upstream classes.

    Idempotent: each patch flags the method it installs and returns early on a
    second call.

    Note on the one dropped patch: the vendored tree also stripped
    ``channels_last_3d`` (a cuDNN-only layout) out of
    ``model/video_vae/memory_efficient_decode.py``. That module is an *opt-in*
    ``ModuleOps`` (``MEMORY_EFFICIENT_DECODE``) that ``VideoDecoderConfigurator``
    does not install, so nothing this loader builds ever reaches it. It is left
    unpatched rather than carrying a runtime patch for dead code; re-derive it
    from the pinned source if the memory-efficient decode path is ever enabled.
    """
    from ltx_core.model.audio_vae.vocoder import VocoderWithBWE, _STFTFn
    from ltx_core.model.transformer.transformer_args import (
        TransformerArgsPreprocessor,
    )
    from ltx_core.model.video_vae.video_vae import VideoDecoder

    _patch_method_source(_STFTFn, "forward", _STFT_OLD, _STFT_NEW, "_tt_unfold_stft")
    _patch_method_source(
        VocoderWithBWE, "forward", _VOCODER_OLD, _VOCODER_NEW, "_tt_compute_dtype"
    )
    _patch_method_source(
        TransformerArgsPreprocessor,
        "_prepare_positional_embeddings",
        _ROPE_OLD,
        _ROPE_NEW,
        "_tt_torch_freq_grid",
    )
    _patch_method_source(
        VideoDecoder,
        "forward",
        _DECODER_DTYPE_OLD,
        _DECODER_DTYPE_NEW,
        "_tt_weights_dtype",
    )


def _bootstrap_ltx_core():
    """Point ``ltx_core`` at the submodule, stub torchaudio, apply the patches."""
    pkg_dir = _ltx_core_pkg_dir()
    if pkg_dir is None:
        raise ImportError(_SUBMODULE_HINT)

    importlib.invalidate_caches()
    _register_bare_package("ltx_core", pkg_dir)
    sys.modules.setdefault("torchaudio", _TorchaudioStub("torchaudio"))


_bootstrap_ltx_core()

from ltx_core.guidance.perturbations import BatchedPerturbationConfig  # noqa: E402
from ltx_core.model.transformer.modality import Modality  # noqa: E402
from ltx_core.model.transformer.model_configurator import (  # noqa: E402
    LTXModelConfigurator,
)
from ltx_core.model.video_vae.model_configurator import (  # noqa: E402
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)
from ltx_core.model.audio_vae.model_configurator import (  # noqa: E402
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)

_apply_runtime_patches()

_HF_REPO = "Lightricks/LTX-2.3"

# Cached Pro/dev checkpoint (43GB) holding real weights for every in-file
# component. The video-VAE variants copy their ``vae.{decoder,encoder}.*`` +
# ``vae.per_channel_statistics.*`` tensors out of it (lazily, via safe_open) so
# the single-device VAE tests run against REAL weights. If the path is absent
# (e.g. CI without the cache mount) the VAE falls back to random init.
_CHECKPOINT_PATH = (
    "/proj_sw/user_dev/dnikolic/model_cache/ltx-checkpoints/"
    "ltx-2.3-22b-dev.safetensors"
)

# ── Embedded transformer config ─────────────────────────────────────────────
# Extracted from the LTX-2.3 22B checkpoint's safetensors header (the
# "transformer" sub-dict of the model config). ``LTXModelConfigurator.from_config``
# reads ONLY ``config["transformer"]`` (both directly and via
# ``_build_caption_projections``), so the full dict here just nests that sub-dict
# under the "transformer" key. ``caption_proj_before_connector=True`` puts the
# caption projection in the text encoder (22B path), so no projection module is
# built inside the transformer -- the cross-attention context arrives already at
# ``cross_attention_dim``.
_TRANSFORMER_CONFIG = {
    "_class_name": "AVTransformer3DModel",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 128,
    "attention_type": "default",
    "caption_channels": 3840,
    "cross_attention_dim": 4096,
    "double_self_attention": False,
    "dropout": 0.0,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
    "num_attention_heads": 32,
    "num_embeds_ada_norm": 1000,
    "num_layers": 48,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "cross_attention_norm": True,
    "out_channels": 128,
    "upcast_attention": False,
    "use_linear_projection": False,
    "qk_norm": "rms_norm",
    "standardization_norm": "rms_norm",
    "positional_embedding_type": "rope",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000,
    "av_ca_timestep_scale_multiplier": 1000.0,
    "causal_temporal_positioning": True,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "use_audio_video_cross_attention": True,
    "share_ff": False,
    "audio_out_channels": 128,
    "audio_cross_attention_dim": 2048,
    "audio_positional_embedding_max_pos": [20],
    "av_cross_ada_norm": True,
    "use_embeddings_connector": True,
    "connector_attention_head_dim": 128,
    "connector_num_attention_heads": 32,
    "connector_num_layers": 8,
    "connector_positional_embedding_max_pos": [4096],
    "connector_num_learnable_registers": 128,
    "connector_norm_output": True,
    "use_middle_indices_grid": True,
    "apply_gated_attention": True,
    "connector_apply_gated_attention": True,
    "caption_projection_first_linear": False,
    "caption_projection_second_linear": False,
    "caption_proj_input_norm": False,
    "connector_learnable_registers_std": 1,
    "caption_proj_before_connector": True,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_attention_heads": 32,
    "cross_attention_adaln": True,
    "text_encoder_norm_type": "per_token_rms",
    "rope_type": "split",
    "frequencies_precision": "float64",
}
_MODEL_CONFIG = {"transformer": _TRANSFORMER_CONFIG}

# ── Embedded video-VAE config ────────────────────────────────────────────────
# The "vae" sub-dict of the LTX-2.3 checkpoint config (CausalVideoAutoencoder).
# ``Video{Decoder,Encoder}Configurator.from_config`` reads ONLY ``config["vae"]``.
# Encoder 318.9M / Decoder 407.2M params -- both fit a single chip. Note
# ``timestep_conditioning`` and ``causal_decoder`` are False for this checkpoint,
# so the decoder forward needs no timestep and injects no noise.
_VAE_CONFIG = {
    "vae": {
        "_class_name": "CausalVideoAutoencoder",
        "dims": 3,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 128,
        "encoder_blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_space_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 6}],
            ["compress_time_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 2}],
            ["compress_all_res", {"multiplier": 1}],
            ["res_x", {"num_layers": 2}],
        ],
        "decoder_blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_space", {"multiplier": 2}],
            ["res_x", {"num_layers": 6}],
            ["compress_time", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all", {"multiplier": 1}],
            ["res_x", {"num_layers": 2}],
            ["compress_all", {"multiplier": 2}],
            ["res_x", {"num_layers": 2}],
        ],
        "scaling_factor": 1.0,
        "norm_layer": "pixel_norm",
        "patch_size": 4,
        "latent_log_var": "uniform",
        "use_quant_conv": False,
        "causal_decoder": False,
        "timestep_conditioning": False,
        "normalize_latent_channels": False,
        "encoder_base_channels": 128,
        "decoder_base_channels": 128,
        "spatial_padding_mode": "zeros",
    }
}

# Video-VAE test shapes. Compression is 8x temporal / 32x spatial:
#   encoder video (B,3,F,H,W)   -> latent (B,128, 1+(F-1)/8, H/32, W/32)
#   decoder latent (B,128,F',H',W') -> video (B,3, 8*(F'-1)+1, 32*H', 32*W')
# The two are exact round-trip inverses at these dims (verified on CPU).
_VAE_LATENT_CHANNELS = 128
_VAE_VIDEO_CHANNELS = 3
_VAE_ENC_VIDEO_SHAPE = (_VAE_VIDEO_CHANNELS, 9, 256, 256)  # -> latent (128,2,8,8)
_VAE_DEC_LATENT_SHAPE = (_VAE_LATENT_CHANNELS, 2, 8, 8)  # -> video (3,9,256,256)

# ── Embedded audio-VAE config ────────────────────────────────────────────────
# The "audio_vae" sub-dict of the checkpoint config (stereo mel autoencoder,
# ch_mult=[1,2,4], z_channels=8, causal on the height/freq axis). The
# Audio{Decoder,Encoder}Configurator read the nested model.params.ddconfig +
# preprocessing.stft/mel. Encoder 21.3M / decoder 31.9M params. torchaudio-based
# preprocessing (AudioProcessor) is NOT on the encode/decode path we trace.
_AUDIO_VAE_CONFIG = {
    "audio_vae": {
        "model": {
            "params": {
                "ddconfig": {
                    "double_z": True,
                    "mel_bins": 64,
                    "z_channels": 8,
                    "resolution": 256,
                    "downsample_time": False,
                    "in_channels": 2,
                    "out_ch": 2,
                    "ch": 128,
                    "ch_mult": [1, 2, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                    "mid_block_add_attention": False,
                    "norm_type": "pixel",
                    "causality_axis": "height",
                },
                "sampling_rate": 16000,
            }
        },
        "preprocessing": {
            "stft": {
                "filter_length": 1024,
                "hop_length": 160,
                "win_length": 1024,
                "causal": True,
            },
            "mel": {"n_mel_channels": 64, "mel_fmin": 0, "mel_fmax": 8000},
        },
    }
}

# ── Embedded vocoder config ──────────────────────────────────────────────────
# BigVGAN-v2-style vocoder + band-width-extension (BWE) stage -> VocoderWithBWE
# (128.5M params). Runs its forward in fp32 (autocast) through ~108 convs plus an
# internal STFT. The COMFY key filter uses a kv-op (strip one 'vocoder.' prefix).
_VOCODER_CONFIG = {
    "vocoder": {
        "vocoder": {
            "upsample_initial_channel": 1536,
            "resblock": "AMP1",
            "upsample_rates": [5, 2, 2, 2, 2, 2],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "stereo": True,
            "use_tanh_at_final": False,
            "activation": "snakebeta",
            "use_bias_at_final": False,
        },
        "bwe": {
            "upsample_initial_channel": 512,
            "resblock": "AMP1",
            "upsample_rates": [6, 5, 2, 2, 2],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_kernel_sizes": [12, 11, 4, 4, 4],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "stereo": True,
            "use_tanh_at_final": False,
            "activation": "snakebeta",
            "use_bias_at_final": False,
            "apply_final_activation": False,
            "input_sampling_rate": 16000,
            "output_sampling_rate": 48000,
            "hop_length": 80,
            "n_fft": 512,
            "win_size": 512,
            "num_mels": 64,
        },
    }
}

# Audio test shapes (verified on CPU with real weights):
#   audio encoder spectrogram (B, 2, 256, 64)   -> latent (B, 8, 64, 16)
#   audio decoder latent      (B, 8, 64, 16)     -> spectrogram (B, 2, 253, 64)
#   vocoder mel               (B, 2, 64, 64)     -> waveform (B, 2, 30720)
_AUDIO_ENC_SPEC_SHAPE = (2, 256, 64)
_AUDIO_DEC_LATENT_SHAPE = (8, 64, 16)
_VOCODER_MEL_SHAPE = (2, 64, 64)

# ── Derived feature dims (read off _TRANSFORMER_CONFIG) ──────────────────────
# model.py: inner_dim = num_attention_heads * attention_head_dim. The cross-attn
# context (attn2.context_dim) == cross_attention_dim. transformer_args.prepare
# reshapes context to (B, -1, inner_dim); inner_dim == cross_attention_dim here,
# so 4096 video / 2048 audio is consistent. (modality.py / transformer.py /
# transformer_args.py.)
_IN_CHANNELS = _TRANSFORMER_CONFIG["in_channels"]  # latent feature dim D = 128
_AUDIO_IN_CHANNELS = 128  # audio_in_channels default (model.py)
_VIDEO_CTX_DIM = (
    _TRANSFORMER_CONFIG["num_attention_heads"]
    * _TRANSFORMER_CONFIG["attention_head_dim"]
)  # 4096 == cross_attention_dim
_AUDIO_CTX_DIM = (
    _TRANSFORMER_CONFIG["audio_num_attention_heads"]
    * _TRANSFORMER_CONFIG["audio_attention_head_dim"]
)  # 2048 == audio_cross_attention_dim

# Minimal valid sequence dims for a reduced-layer CPU sanity forward.
_VIDEO_TOKENS = 4
_AUDIO_TOKENS = 4
_CTX_SEQ = 8

# variant -> intended checkpoint filename (NOT loaded by this scaffold).
_VARIANT_CHECKPOINT = {
    "Fast": "ltx-2.3-22b-distilled-1.1.safetensors",
    "Pro": "ltx-2.3-22b-dev.safetensors",
}


class ModelVariant(StrEnum):
    # Transformer product tiers (same architecture, different checkpoint).
    LTX2_3_FAST = "Fast"
    LTX2_3_PRO = "Pro"
    # Video-VAE components (single-device, real weights from the cached ckpt).
    VIDEO_VAE_DECODER = "VideoVaeDecoder"
    VIDEO_VAE_ENCODER = "VideoVaeEncoder"
    # Audio-VAE + vocoder components (single-device, real weights).
    AUDIO_VAE_DECODER = "AudioVaeDecoder"
    AUDIO_VAE_ENCODER = "AudioVaeEncoder"
    VOCODER = "Vocoder"


_TRANSFORMER_VARIANTS = (ModelVariant.LTX2_3_FAST, ModelVariant.LTX2_3_PRO)
_VIDEO_VAE_VARIANTS = (ModelVariant.VIDEO_VAE_DECODER, ModelVariant.VIDEO_VAE_ENCODER)


# ── Tensors-only wrapper ─────────────────────────────────────────────────────
class _LTXModelWrapper(torch.nn.Module):
    """Wrap the native ``LTXModel`` (which takes ``Modality`` objects) in a
    plain-tensor ``forward`` so the bringup harness can trace it.

    The non-tensor structural argument (the no-op perturbation config) is built
    inside ``forward`` from the batch size; only tensors cross the boundary.
    Returns the video output tensor (the audio output is computed but dropped).
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        video_latent,
        video_sigma,
        video_timesteps,
        video_positions,
        video_context,
        audio_latent,
        audio_sigma,
        audio_timesteps,
        audio_positions,
        audio_context,
    ):
        video = Modality(
            latent=video_latent,
            sigma=video_sigma,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
        )
        audio = Modality(
            latent=audio_latent,
            sigma=audio_sigma,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_context,
        )
        perturbations = BatchedPerturbationConfig.empty(video_latent.shape[0])
        video_out, _audio_out = self.model(video, audio, perturbations)
        return video_out


class _VideoDecoderWrapper(torch.nn.Module):
    """Tensors-only forward for the video-VAE decoder: latent -> video.

    The native ``VideoDecoder.forward`` takes optional ``timestep`` / ``generator``
    kwargs, but this checkpoint has ``timestep_conditioning=False`` so a single
    latent tensor is the only input that crosses the trace boundary.
    """

    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent):
        return self.decoder(latent)


class _VideoEncoderWrapper(torch.nn.Module):
    """Tensors-only forward for the video-VAE encoder: video -> latent."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, video):
        out = self.encoder(video)
        return out[0] if isinstance(out, (tuple, list)) else out


class _TensorForwardWrapper(torch.nn.Module):
    """Tensors-only forward for a component whose native ``forward`` already takes
    and returns a single tensor (audio VAE decoder/encoder, vocoder)."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        out = self.module(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def _load_vocoder_weights(module: torch.nn.Module) -> bool:
    """Load real vocoder weights via the kv-op key filter.

    The vocoder filter strips exactly one leading ``vocoder.`` prefix through a
    key-value operation (``apply_to_key_value``), which ``apply_to_key`` does not
    apply — so this path pre-filters to ``vocoder.*`` keys and runs the kv-op.
    """
    if not os.path.exists(_CHECKPOINT_PATH):
        return False

    from safetensors import safe_open

    remapped = {}
    with safe_open(_CHECKPOINT_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith("vocoder."):
                continue
            v = f.get_tensor(k)
            for res in VOCODER_COMFY_KEYS_FILTER.apply_to_key_value(k, v):
                remapped[res.new_key] = res.new_value

    result = module.load_state_dict(remapped, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Vocoder weight load mismatch: missing={result.missing_keys[:5]} "
            f"unexpected={result.unexpected_keys[:5]}"
        )
    return True


def _load_vae_weights(module: torch.nn.Module, key_filter) -> bool:
    """Copy the matching VAE tensors from the cached checkpoint into ``module``.

    Returns True if real weights were loaded, False if the checkpoint is absent
    (module keeps its random init). Uses ``safe_open`` so only the ~86 matched
    tensors are materialized, not the full 43GB file.
    """
    if not os.path.exists(_CHECKPOINT_PATH):
        return False

    from safetensors import safe_open  # local import: optional at scaffold time

    remapped = {}
    with safe_open(_CHECKPOINT_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            new_k = key_filter.apply_to_key(k)
            if new_k is None:
                continue
            remapped[new_k] = f.get_tensor(k)

    result = module.load_state_dict(remapped, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"VAE weight load mismatch: missing={result.missing_keys[:5]} "
            f"unexpected={result.unexpected_keys[:5]}"
        )
    return True


class ModelLoader(ForgeModel):
    """LTX-2.3 22B audio-video DiT transformer loader (Fast / Pro variants)."""

    _VARIANTS = {v: ModelConfig(pretrained_model_name=_HF_REPO) for v in ModelVariant}
    DEFAULT_VARIANT = ModelVariant.LTX2_3_FAST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX2_3",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, num_layers=None, **kwargs):
        """Build one LTX-2.3 component and wrap it for a tensors-only forward.

        Transformer variants (Fast/Pro) build the native ``LTXModel`` from the
        embedded config with RANDOM weights -- the full 48-layer model is ~19B
        params and is weight-bound on a single chip (multichip TP follow-up).
        ``num_layers`` overrides layer count for CPU sanity checks only.

        Video-VAE variants build the decoder/encoder from ``_VAE_CONFIG`` and load
        REAL weights from the cached checkpoint (random fallback if absent).
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._variant in (
            ModelVariant.VIDEO_VAE_DECODER,
            ModelVariant.VIDEO_VAE_ENCODER,
        ):
            # Build in fp32, load real weights, then cast to the target dtype so
            # the checkpoint values are copied cleanly before down-casting.
            if self._variant == ModelVariant.VIDEO_VAE_DECODER:
                base = VideoDecoderConfigurator.from_config(_VAE_CONFIG).eval()
                _load_vae_weights(base, VAE_DECODER_COMFY_KEYS_FILTER)
                base = base.to(dtype)
                # Stamp the weight dtype so the decoder forward skips the
                # non-traceable ``next(self.parameters())`` dtype lookup.
                base._forge_weights_dtype = dtype
                self.model = _VideoDecoderWrapper(base)
            else:
                base = VideoEncoderConfigurator.from_config(_VAE_CONFIG).eval()
                _load_vae_weights(base, VAE_ENCODER_COMFY_KEYS_FILTER)
                self.model = _VideoEncoderWrapper(base)
            self.model = self.model.to(dtype)
            return self.model

        if self._variant in (
            ModelVariant.AUDIO_VAE_DECODER,
            ModelVariant.AUDIO_VAE_ENCODER,
            ModelVariant.VOCODER,
        ):
            if self._variant == ModelVariant.AUDIO_VAE_DECODER:
                base = AudioDecoderConfigurator.from_config(_AUDIO_VAE_CONFIG).eval()
                _load_vae_weights(base, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER)
            elif self._variant == ModelVariant.AUDIO_VAE_ENCODER:
                base = AudioEncoderConfigurator.from_config(_AUDIO_VAE_CONFIG).eval()
                _load_vae_weights(base, AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER)
            else:  # VOCODER
                base = VocoderConfigurator.from_config(_VOCODER_CONFIG).eval()
                _load_vocoder_weights(base)
                # The vocoder (BigVGAN-v2 + BWE) upstream runs its whole forward in
                # fp32 (``mel_spec.float()`` under ``autocast(dtype=float32)``)
                # because bf16 accumulation over 108 sequential convs degrades
                # spectral metrics. Two problems on TT: (1) CPU autocast-to-fp32 is
                # a no-op, so with bf16 weights the fp32 activations hit "Input type
                # (float) and bias type (BFloat16) should be the same"; (2) the fp32
                # activation footprint overflows device L1 on the conv stack. So we
                # run the component in bf16 end-to-end -- stamp a compute-dtype flag
                # that the patched forward reads to skip the fp32 upcast, and keep
                # weights bf16. PCC is then measured CPU-vs-device at bf16.
                base = base.to(dtype)
                base._forge_compute_dtype = dtype
                self.model = _TensorForwardWrapper(base)
                return self.model
            base = base.to(dtype)
            self.model = _TensorForwardWrapper(base)
            return self.model

        # ── transformer (Fast / Pro) ─────────────────────────────────────────
        config = _MODEL_CONFIG
        if num_layers is not None:
            config = {"transformer": {**_TRANSFORMER_CONFIG, "num_layers": num_layers}}

        base = LTXModelConfigurator.from_config(config)
        base = base.to(dtype).eval()

        self.model = _LTXModelWrapper(base)
        if dtype_override is not None:
            self.model = self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Synthetic plain tensors at minimal valid shapes, returned in the
        wrapper's forward-arg order (video block then audio block).

        Shapes follow ``Modality`` (modality.py) + the args preprocessors
        (transformer_args.py): latent (B, T, D=in_channels); context
        (B, ctx_seq, inner_dim) where inner_dim == heads*head_dim == the cross-
        attention dim (4096 video / 2048 audio, caption projection lives in the
        text encoder for 22B); positions (B, n_pos_dims, T, 2) with n_pos_dims=3
        video / 1 audio and last dim = [start, end) patch bounds because
        use_middle_indices_grid=True; timesteps (B, T); sigma (B,).
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._variant == ModelVariant.VIDEO_VAE_DECODER:
            return [torch.randn(batch_size, *_VAE_DEC_LATENT_SHAPE, dtype=dtype)]
        if self._variant == ModelVariant.VIDEO_VAE_ENCODER:
            return [torch.randn(batch_size, *_VAE_ENC_VIDEO_SHAPE, dtype=dtype)]
        if self._variant == ModelVariant.AUDIO_VAE_DECODER:
            return [torch.randn(batch_size, *_AUDIO_DEC_LATENT_SHAPE, dtype=dtype)]
        if self._variant == ModelVariant.AUDIO_VAE_ENCODER:
            return [torch.randn(batch_size, *_AUDIO_ENC_SPEC_SHAPE, dtype=dtype)]
        if self._variant == ModelVariant.VOCODER:
            return [torch.randn(batch_size, *_VOCODER_MEL_SHAPE, dtype=dtype)]

        def _positions(n_pos_dims, tokens):
            # [start, end) integer patch bounds: end = start + 1.
            start = torch.arange(tokens, dtype=dtype).view(1, 1, tokens, 1)
            start = start.expand(batch_size, n_pos_dims, tokens, 1)
            return torch.cat([start, start + 1], dim=-1)

        return [
            # video
            torch.randn(batch_size, _VIDEO_TOKENS, _IN_CHANNELS, dtype=dtype),
            torch.full((batch_size,), 0.5, dtype=dtype),
            torch.full((batch_size, _VIDEO_TOKENS), 0.5, dtype=dtype),
            _positions(3, _VIDEO_TOKENS),
            torch.randn(batch_size, _CTX_SEQ, _VIDEO_CTX_DIM, dtype=dtype),
            # audio
            torch.randn(batch_size, _AUDIO_TOKENS, _AUDIO_IN_CHANNELS, dtype=dtype),
            torch.full((batch_size,), 0.5, dtype=dtype),
            torch.full((batch_size, _AUDIO_TOKENS), 0.5, dtype=dtype),
            _positions(1, _AUDIO_TOKENS),
            torch.randn(batch_size, _CTX_SEQ, _AUDIO_CTX_DIM, dtype=dtype),
        ]

    def unpack_forward_output(self, output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    # ── Multichip tensor-parallel plan (Megatron 1D on the model axis) ──────
    def get_mesh_config(self, num_devices: int):
        """Return ((1, num_devices), ("batch", "model")) for Megatron-style TP."""
        return (1, num_devices), ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP map over the transformer blocks. Non-sharded dim is
        ``None`` (replicated).

        Module names are taken from the upstream ``BasicAVTransformerBlock``
        (transformer.py): per-block attentions ``attn1`` / ``attn2`` (video),
        ``audio_attn1`` / ``audio_attn2`` (audio), and the AV cross-attentions
        ``audio_to_video_attn`` / ``video_to_audio_attn``; feed-forwards ``ff``
        / ``audio_ff``. Each ``Attention`` exposes ``to_q`` / ``to_k`` / ``to_v``
        and an output projection; ``FeedForward`` wraps an ``nn.Sequential``
        ``net``. The exact submodule names of ``Attention`` / ``FeedForward``
        were not fully inspected, so this is written DEFENSIVELY: any missing
        attribute is skipped. Column-parallel q/k/v + row-parallel out is the
        standard Megatron split.
        """
        shard_specs = {}
        wrapped = getattr(model, "model", model)
        blocks = getattr(wrapped, "transformer_blocks", None)
        if blocks is None:
            return shard_specs

        attn_names = (
            "attn1",
            "attn2",
            "audio_attn1",
            "audio_attn2",
            "audio_to_video_attn",
            "video_to_audio_attn",
        )
        ff_names = ("ff", "audio_ff")

        def _w(module, attr):
            sub = getattr(module, attr, None)
            return getattr(sub, "weight", None) if sub is not None else None

        for block in blocks:
            for attn_name in attn_names:
                attn = getattr(block, attn_name, None)
                if attn is None:
                    continue
                # Column-parallel q/k/v projections. ``to_gate_logits`` is a
                # per-head gate (out dim == heads, verified shape (heads, dim));
                # its output is applied per-head to the head-sharded attn output
                # (see ops.PytorchGatedAttention), so it MUST be sharded on the
                # head/output dim to match — a replicated gate shape-mismatches.
                for proj in ("to_q", "to_k", "to_v", "to_gate_logits"):
                    w = _w(attn, proj)
                    if w is not None:
                        shard_specs[w] = ("model", None)
                # Row-parallel output projection. ltx_core's Attention may expose
                # the output projection under one of these names; try each.
                for out_name in ("to_out", "out_proj", "proj_out"):
                    out = getattr(attn, out_name, None)
                    if out is None:
                        continue
                    # to_out is sometimes an nn.Sequential/ModuleList.
                    if hasattr(out, "weight"):
                        shard_specs[out.weight] = (None, "model")
                    elif hasattr(out, "__getitem__"):
                        try:
                            shard_specs[out[0].weight] = (None, "model")
                        except (IndexError, AttributeError, TypeError):
                            pass
                    break
            for ff_name in ff_names:
                ff = getattr(block, ff_name, None)
                if ff is None or not hasattr(ff, "net"):
                    continue
                net = ff.net
                # net[0] (or its .proj) is the up-projection (column); net[-1] is
                # the down-projection (row).
                first = net[0]
                first_w = getattr(getattr(first, "proj", first), "weight", None)
                if first_w is not None:
                    shard_specs[first_w] = ("model", None)
                last_w = getattr(net[-1], "weight", None)
                if last_w is not None:
                    shard_specs[last_w] = (None, "model")
        return shard_specs
