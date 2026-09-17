# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 pipeline component loader for text-to-video (+audio) generation.

LTX-2 (Lightricks) loads as a ``LTX2Pipeline`` whose ``model_index.json``
enumerates six compilable sub-models. Each is exposed here as its own
``ModelVariant`` with a tensors-only wrapper, so every component can be brought
up independently on a single device (small ones) or a tensor-parallel mesh
(the weight-bound transformer / text encoder).

Repository: https://github.com/Lightricks/LTX-2
Weights:    https://huggingface.co/Lightricks/LTX-2

Components (params measured by random-init unless noted "est"):

    transformer  LTX2VideoTransformer3DModel    ~19B (est)  diffusers
    text_encoder Gemma3ForConditionalGeneration ~12B (est)  transformers
    connectors   LTX2TextConnectors             1.43B       diffusers.pipelines.ltx2
    vae          AutoencoderKLLTX2Video         1.22B       diffusers
    vocoder      LTX2Vocoder                    0.056B      diffusers.pipelines.ltx2
    audio_vae    AutoencoderKLLTX2Audio         0.053B      diffusers

The ``model_index.json`` module name ``ltx2`` resolves to
``diffusers.pipelines.ltx2`` (NOT an external package); ``LTX2TextConnectors``
and ``LTX2Vocoder`` are ``ModelMixin``/``ConfigMixin`` so ``from_pretrained``
with ``subfolder=`` works for every component.
"""
from typing import Optional

import torch
from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2TextConnectors, LTX2Vocoder
from transformers import Gemma3ForConditionalGeneration

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


def _xla_apply_split_rotary_emb(x, freqs):
    """XLA-safe replacement for diffusers ``apply_split_rotary_emb``.

    Upstream applies rope in ``(b, h, t, d)`` layout and merges the heads back
    with ``out.swapaxes(1, 2).reshape(b, t, -1)``. On XLA that reshape is traced
    as a strict ``aten.view``: XLA tensors report dense row-major strides, so
    ``reshape`` takes the view path, but the later decomposition runs on fake
    tensors that *do* carry the transposed strides, and the view is rejected —
    ``ValueError: Cannot view a tensor with shape [1, 128, 30, 128] and strides
    (491520, 128, 16384, 1) as a tensor with shape (1, 128, 3840)``.
    ``.contiguous()`` cannot fix it (a no-op on XLA lazy tensors).

    This version keeps ``x`` in ``(b, t, h, d)`` layout and transposes ``cos`` /
    ``sin`` instead. Those only ever feed elementwise broadcasts, never a
    reshape, so no view-of-transposed op is produced and the head merge is a
    plain contiguous reshape. Bit-exact with upstream on CPU in fp32 and bf16.

    See https://github.com/tenstorrent/tt-xla/issues/5196.
    """
    cos, sin = freqs

    x_dtype = x.dtype
    transposed_freqs = x.ndim != 4 and cos.ndim == 4
    if transposed_freqs:
        # cos is (b, h, t, r) -> view x as (b, t, h, dim_per_head), no transpose
        b, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1)
        cos = cos.swapaxes(1, 2)  # (b, t, h, r)
        sin = sin.swapaxes(1, 2)

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(
            f"Expected x.shape[-1] to be even for split rotary, got {last}."
        )
    r = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, r).float()
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]

    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    if transposed_freqs:
        out = out.reshape(b, t, h * last)
    else:
        out = out.reshape(*out.shape[:-2], last)

    return out.to(dtype=x_dtype)


def _patch_rope_for_xla():
    """Install the XLA-safe rope head-merge (tt-xla#5196).

    Both the transformer's attention processors and the text connectors resolve
    ``apply_split_rotary_emb`` as a module global at call time, so replacing the
    module attribute covers every call site.
    """
    from diffusers.models.transformers import transformer_ltx2

    transformer_ltx2.apply_split_rotary_emb = _xla_apply_split_rotary_emb


_HF_REPO = "Lightricks/LTX-2"

# NOTE: transformer + connectors run split rope, whose upstream head merge
# (`out.swapaxes(1, 2).reshape(b, t, -1)`) cannot be lowered on XLA — see
# tt-xla#5196. `_patch_rope_for_xla()` (above) installs an XLA-safe replacement
# and is applied for both variants in `load_model()`.

# ── Shape constants (minimal valid dims for TT bringup) ─────────────────────
# VAE compression ratios — LTX-2 family (LTX2Pipeline defaults)
_VAE_SPATIAL_COMPRESSION = 32
_VAE_TEMPORAL_COMPRESSION = 8
DEFAULT_NUM_FRAMES = 9
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_FRAME_RATE = 24.0
DEFAULT_SEQ_LEN = 128

# Transformer feature dims (transformer/config.json)
_VIDEO_IN_CHANNELS = 128
_AUDIO_IN_CHANNELS = 128
_CAPTION_CHANNELS = 3840  # == Gemma-3-12B hidden_size
_TEXT_PROJ_IN_FACTOR = 49  # connectors: num_hidden_layers (48) + 1 embedding layer

# Text encoder (Gemma3-12B text_config)
_TE_VOCAB_SIZE = 262208

# Audio VAE / vocoder mel dims (audio_vae/config.json)
_AUDIO_MEL_BINS = 64
_AUDIO_CHANNELS = 2
_AUDIO_VAE_TIME = 64  # minimal mel time frames for a valid audio-VAE forward
_VOCODER_TIME = 16  # minimal mel time frames for a valid vocoder forward

# Latent channel counts (vae/config.json, audio_vae/config.json). The encoder
# emits 2x these (mean ++ logvar moments); the decoder consumes 1x.
_VAE_LATENT_CHANNELS = 128
_AUDIO_VAE_LATENT_CHANNELS = 8

# Video sample dims for the VAE variants. The 32x spatial / 8x temporal
# compression maps a (1, 3, 1, 64, 64) sample to a (1, 128, 1, 2, 2) latent.
_VAE_SAMPLE_FRAMES = 1
_VAE_SAMPLE_SIZE = 64
_VAE_LATENT_FRAMES = 1
_VAE_LATENT_SIZE = _VAE_SAMPLE_SIZE // _VAE_SPATIAL_COMPRESSION

# Audio latent dims: a (1, 2, 64, 64) mel sample encodes to (1, 8, 16, 16).
_AUDIO_VAE_LATENT_MEL = 16
_AUDIO_VAE_LATENT_TIME = 16

# Captured per-component I/O spec (forward arg order == load_inputs() order).
# Validated by random-init CPU forward unless tagged derived.
_COMPONENT_IO_SPEC = {
    "transformer": {  # derived (19B; not CPU-instantiated)
        "inputs": [
            ("hidden_states", "float", (1, 2, _VIDEO_IN_CHANNELS)),
            ("audio_hidden_states", "float", (1, 9, _AUDIO_IN_CHANNELS)),
            ("encoder_hidden_states", "float", (1, DEFAULT_SEQ_LEN, _CAPTION_CHANNELS)),
            (
                "audio_encoder_hidden_states",
                "float",
                (1, DEFAULT_SEQ_LEN, _CAPTION_CHANNELS),
            ),
            # One timestep per batch element, as LTX2Pipeline passes it
            # (t.expand(batch)). The model views the embedding as
            # (batch, -1, dim) so it broadcasts over both the video and the
            # audio token axes; a per-video-token timestep would give
            # temb_audio a video-length token axis and fail to broadcast
            # against the 144 audio tokens.
            ("timestep", "float", (1,)),
            ("encoder_attention_mask", "float", (1, DEFAULT_SEQ_LEN)),
            ("audio_encoder_attention_mask", "float", (1, DEFAULT_SEQ_LEN)),
        ],
        "output": "out[0]",
    },
    "text_encoder": {  # derived (12B; not CPU-instantiated)
        "inputs": [
            ("input_ids", "int", (1, DEFAULT_SEQ_LEN)),
            ("attention_mask", "int", (1, DEFAULT_SEQ_LEN)),
        ],
        "output": "stack(hidden_states) -> (1, seq, 3840, 49)",
    },
    "connectors": {  # validated 1.43B
        "inputs": [
            (
                "text_encoder_hidden_states",
                "float",
                (1, DEFAULT_SEQ_LEN, _CAPTION_CHANNELS, _TEXT_PROJ_IN_FACTOR),
            ),
            ("attention_mask", "int", (1, DEFAULT_SEQ_LEN)),
        ],
        "output": "out[0] -> (1, 128, 3840)",
    },
    "vae": {  # validated 1.22B (encoder 0.694B + decoder 0.528B, round trip)
        "inputs": [("sample", "float", (1, 3, 1, 64, 64))],
        "output": "out.sample -> (1, 3, 1, 64, 64)",
    },
    "vae_encoder": {  # validated 0.694B
        "inputs": [("sample", "float", (1, 3, 1, 64, 64))],
        "output": "latent_dist.parameters -> (1, 256, 1, 2, 2)",
    },
    "vae_decoder": {  # validated 0.528B
        "inputs": [
            (
                "latent",
                "float",
                (1, _VAE_LATENT_CHANNELS, 1, 2, 2),
            )
        ],
        "output": "out.sample -> (1, 3, 1, 64, 64)",
    },
    "audio_vae": {  # validated 0.053B (round trip)
        "inputs": [
            ("sample", "float", (1, _AUDIO_CHANNELS, _AUDIO_MEL_BINS, _AUDIO_VAE_TIME))
        ],
        "output": "out.sample -> (1, 2, 61, 64)",
    },
    "audio_vae_encoder": {  # validated
        "inputs": [
            ("sample", "float", (1, _AUDIO_CHANNELS, _AUDIO_MEL_BINS, _AUDIO_VAE_TIME))
        ],
        "output": "latent_dist.parameters -> (1, 16, 16, 16)",
    },
    "audio_vae_decoder": {  # validated
        "inputs": [
            (
                "latent",
                "float",
                (1, _AUDIO_VAE_LATENT_CHANNELS, 16, 16),
            )
        ],
        "output": "out.sample -> (1, 2, 61, 64)",
    },
    "vocoder": {  # validated 0.056B -> (1, 2, 3840)
        "inputs": [
            (
                "hidden_states",
                "float",
                (1, _AUDIO_CHANNELS, _VOCODER_TIME, _AUDIO_MEL_BINS),
            )
        ],
        "output": "tensor",
    },
}


class ModelVariant(StrEnum):
    LTX2_TRANSFORMER = "Transformer"
    LTX2_TEXT_ENCODER = "TextEncoder"
    LTX2_CONNECTORS = "Connectors"
    # The plain `Vae` / `AudioVae` variants run the full autoencode round trip
    # (encode -> posterior mode -> decode) in one graph, which is what the
    # pipeline's `forward` does. The `*Encoder` / `*Decoder` variants isolate a
    # single direction so each half can be brought up and measured on its own.
    LTX2_VAE = "Vae"
    LTX2_VAE_ENCODER = "VaeEncoder"
    LTX2_VAE_DECODER = "VaeDecoder"
    LTX2_AUDIO_VAE = "AudioVae"
    LTX2_AUDIO_VAE_ENCODER = "AudioVaeEncoder"
    LTX2_AUDIO_VAE_DECODER = "AudioVaeDecoder"
    LTX2_VOCODER = "Vocoder"


# variant -> (component class, model_index subfolder)
_COMPONENT = {
    ModelVariant.LTX2_TRANSFORMER: (LTX2VideoTransformer3DModel, "transformer"),
    ModelVariant.LTX2_TEXT_ENCODER: (Gemma3ForConditionalGeneration, "text_encoder"),
    ModelVariant.LTX2_CONNECTORS: (LTX2TextConnectors, "connectors"),
    ModelVariant.LTX2_VAE: (AutoencoderKLLTX2Video, "vae"),
    ModelVariant.LTX2_VAE_ENCODER: (AutoencoderKLLTX2Video, "vae"),
    ModelVariant.LTX2_VAE_DECODER: (AutoencoderKLLTX2Video, "vae"),
    ModelVariant.LTX2_AUDIO_VAE: (AutoencoderKLLTX2Audio, "audio_vae"),
    ModelVariant.LTX2_AUDIO_VAE_ENCODER: (AutoencoderKLLTX2Audio, "audio_vae"),
    ModelVariant.LTX2_AUDIO_VAE_DECODER: (AutoencoderKLLTX2Audio, "audio_vae"),
    ModelVariant.LTX2_VOCODER: (LTX2Vocoder, "vocoder"),
}


# ── Tensors-only wrappers (pin non-tensor structural args, unwrap outputs) ──
class _TransformerWrapper(torch.nn.Module):
    def __init__(self, transformer, num_frames, height, width, audio_num_frames):
        super().__init__()
        self.transformer = transformer
        self._nf, self._h, self._w = num_frames, height, width
        self._anf = audio_num_frames

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        audio_encoder_attention_mask,
    ):
        out = self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=self._nf,
            height=self._h,
            width=self._w,
            fps=DEFAULT_FRAME_RATE,
            audio_num_frames=self._anf,
            return_dict=False,
        )
        return out[0]


class _TextEncoderWrapper(torch.nn.Module):
    """Gemma3 text encoder -> per-layer hidden states stacked as the connector
    consumes them: (batch, seq, caption_channels, num_layers+1)."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # tuple of (num_layers+1) tensors, each (batch, seq, hidden)
        return torch.stack(out.hidden_states, dim=-1)


class _ConnectorsWrapper(torch.nn.Module):
    def __init__(self, connectors):
        super().__init__()
        self.connectors = connectors

    def forward(self, text_encoder_hidden_states, attention_mask):
        out = self.connectors(
            text_encoder_hidden_states=text_encoder_hidden_states,
            attention_mask=attention_mask,
            padding_side="left",
            scale_factor=8,
        )
        return out[0]  # video text embedding


class _VaeWrapper(torch.nn.Module):
    """Full autoencode round trip: encode -> posterior mode -> decode.

    Deterministic (``sample_posterior=False``), so both halves of the VAE are
    exercised in a single graph. Use ``_VaeEncoderWrapper`` /
    ``_VaeDecoderWrapper`` to compile one direction on its own.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, sample):
        out = self.vae(sample, return_dict=True)
        return out.sample


class _VaeEncoderWrapper(torch.nn.Module):
    """Encoder half only. Returns the raw Gaussian moments (mean ++ logvar) so
    the full encoder output is covered by the comparison, not just the mean."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, sample):
        return self.vae.encode(sample, return_dict=True).latent_dist.parameters


class _VaeDecoderWrapper(torch.nn.Module):
    """Decoder half only. ``temb`` is None because the checkpoint sets
    ``timestep_conditioning=False``."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=True).sample


class _AudioVaeWrapper(torch.nn.Module):
    """Full audio autoencode round trip (deterministic)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, sample):
        out = self.vae(sample, return_dict=True)
        return out.sample


class _AudioVaeEncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, sample):
        return self.vae.encode(sample, return_dict=True).latent_dist.parameters


class _AudioVaeDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=True).sample


class _VocoderWrapper(torch.nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.vocoder = vocoder

    def forward(self, hidden_states):
        return self.vocoder(hidden_states, time_last=False)


# Variant -> wrapper. The transformer is built separately (its wrapper needs the
# derived latent dims), so it is intentionally absent here.
_WRAPPER = {
    ModelVariant.LTX2_TEXT_ENCODER: _TextEncoderWrapper,
    ModelVariant.LTX2_CONNECTORS: _ConnectorsWrapper,
    ModelVariant.LTX2_VAE: _VaeWrapper,
    ModelVariant.LTX2_VAE_ENCODER: _VaeEncoderWrapper,
    ModelVariant.LTX2_VAE_DECODER: _VaeDecoderWrapper,
    ModelVariant.LTX2_AUDIO_VAE: _AudioVaeWrapper,
    ModelVariant.LTX2_AUDIO_VAE_ENCODER: _AudioVaeEncoderWrapper,
    ModelVariant.LTX2_AUDIO_VAE_DECODER: _AudioVaeDecoderWrapper,
    ModelVariant.LTX2_VOCODER: _VocoderWrapper,
}


class ModelLoader(ForgeModel):
    """LTX-2 pipeline component loader (one variant per compilable component)."""

    _VARIANTS = {v: ModelConfig(pretrained_model_name=_HF_REPO) for v in ModelVariant}
    DEFAULT_VARIANT = ModelVariant.LTX2_TRANSFORMER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX2",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # ── derived video/audio latent dims (shared by transformer load_inputs) ─
    @staticmethod
    def _latent_dims():
        latent_num_frames = (DEFAULT_NUM_FRAMES - 1) // _VAE_TEMPORAL_COMPRESSION + 1
        latent_height = DEFAULT_HEIGHT // _VAE_SPATIAL_COMPRESSION
        latent_width = DEFAULT_WIDTH // _VAE_SPATIAL_COMPRESSION
        video_tokens = max(1, latent_num_frames * latent_height * latent_width)
        audio_num_frames = 9
        # One audio token per latent frame. LTX2Pipeline._pack_audio_latents is
        # called without patch sizes, so it takes the [B, C, L, M] -> [B, L, C*M]
        # branch: the mel bins are folded into the *feature* axis (C*M == 8*16 ==
        # audio_in_channels == 128), not the sequence axis, and
        # prepare_audio_coords likewise emits num_patches == num_frames. Folding
        # mel bins into the token count instead gives the audio rope a
        # 16x-too-short token axis and the per-head split then mismatches
        # cos/sin ("size of tensor a (512) must match tensor b (32)").
        audio_tokens = audio_num_frames
        return (
            latent_num_frames,
            latent_height,
            latent_width,
            video_tokens,
            audio_num_frames,
            audio_tokens,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load one LTX-2 component (direct from_pretrained, wrapped to a
        tensors-only forward)."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # The transformer and the connectors both run split rope, whose upstream
        # head merge cannot be lowered on XLA (tt-xla#5196).
        if self._variant in (
            ModelVariant.LTX2_TRANSFORMER,
            ModelVariant.LTX2_CONNECTORS,
        ):
            _patch_rope_for_xla()

        cls, subfolder = _COMPONENT[self._variant]
        base = cls.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder=subfolder,
            torch_dtype=dtype,
        ).eval()

        if self._variant == ModelVariant.LTX2_TRANSFORMER:
            lnf, lh, lw, _, anf, _ = self._latent_dims()
            self.model = _TransformerWrapper(base, lnf, lh, lw, anf)
        else:
            wrapper = _WRAPPER.get(self._variant)
            if wrapper is None:  # pragma: no cover
                raise ValueError(f"Unknown variant {self._variant}")
            self.model = wrapper(base)

        if dtype_override is not None:
            self.model = self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Synthetic tensors at captured shapes, returned as a list in the
        wrapper's forward arg order."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        v = self._variant

        if v == ModelVariant.LTX2_TRANSFORMER:
            lnf, lh, lw, vtok, anf, atok = self._latent_dims()
            return [
                torch.randn(batch_size, vtok, _VIDEO_IN_CHANNELS, dtype=dtype),
                torch.randn(batch_size, atok, _AUDIO_IN_CHANNELS, dtype=dtype),
                torch.randn(
                    batch_size, DEFAULT_SEQ_LEN, _CAPTION_CHANNELS, dtype=dtype
                ),
                torch.randn(
                    batch_size, DEFAULT_SEQ_LEN, _CAPTION_CHANNELS, dtype=dtype
                ),
                # (batch,) — see the timestep note in _COMPONENT_IO_SPEC.
                torch.full((batch_size,), 1000.0, dtype=dtype),
                torch.ones(batch_size, DEFAULT_SEQ_LEN, dtype=dtype),
                torch.ones(batch_size, DEFAULT_SEQ_LEN, dtype=dtype),
            ]
        if v == ModelVariant.LTX2_TEXT_ENCODER:
            return [
                torch.randint(
                    0, _TE_VOCAB_SIZE, (batch_size, DEFAULT_SEQ_LEN), dtype=torch.long
                ),
                torch.ones(batch_size, DEFAULT_SEQ_LEN, dtype=torch.long),
            ]
        if v == ModelVariant.LTX2_CONNECTORS:
            return [
                torch.randn(
                    batch_size,
                    DEFAULT_SEQ_LEN,
                    _CAPTION_CHANNELS,
                    _TEXT_PROJ_IN_FACTOR,
                    dtype=dtype,
                ),
                torch.ones(batch_size, DEFAULT_SEQ_LEN, dtype=torch.long),
            ]
        if v in (ModelVariant.LTX2_VAE, ModelVariant.LTX2_VAE_ENCODER):
            return [
                torch.randn(
                    batch_size,
                    3,
                    _VAE_SAMPLE_FRAMES,
                    _VAE_SAMPLE_SIZE,
                    _VAE_SAMPLE_SIZE,
                    dtype=dtype,
                )
            ]
        if v == ModelVariant.LTX2_VAE_DECODER:
            return [
                torch.randn(
                    batch_size,
                    _VAE_LATENT_CHANNELS,
                    _VAE_LATENT_FRAMES,
                    _VAE_LATENT_SIZE,
                    _VAE_LATENT_SIZE,
                    dtype=dtype,
                )
            ]
        if v in (ModelVariant.LTX2_AUDIO_VAE, ModelVariant.LTX2_AUDIO_VAE_ENCODER):
            return [
                torch.randn(
                    batch_size,
                    _AUDIO_CHANNELS,
                    _AUDIO_MEL_BINS,
                    _AUDIO_VAE_TIME,
                    dtype=dtype,
                )
            ]
        if v == ModelVariant.LTX2_AUDIO_VAE_DECODER:
            return [
                torch.randn(
                    batch_size,
                    _AUDIO_VAE_LATENT_CHANNELS,
                    _AUDIO_VAE_LATENT_MEL,
                    _AUDIO_VAE_LATENT_TIME,
                    dtype=dtype,
                )
            ]
        if v == ModelVariant.LTX2_VOCODER:
            return [
                torch.randn(
                    batch_size,
                    _AUDIO_CHANNELS,
                    _VOCODER_TIME,
                    _AUDIO_MEL_BINS,
                    dtype=dtype,
                )
            ]
        raise ValueError(f"Unknown variant {v}")  # pragma: no cover

    def unpack_forward_output(self, output):
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output

    # ── Multichip tensor-parallel plan (Megatron 1D on the model axis) ──────
    def get_mesh_config(self, num_devices: int):
        """Return ((1, num_devices), ("batch", "model")) for Megatron-style TP.

        Only the weight-bound transformer / text encoder are sharded; the small
        components replicate (``load_shard_spec`` returns an empty map for them).
        """
        return (1, num_devices), ("batch", "model")

    @staticmethod
    def _shard_transformer(model):
        """Megatron-style TP map for LTX2VideoTransformer3DModel (per-block
        attention + MLP)."""
        shard_specs = {}
        transformer = getattr(model, "transformer", model)
        for block in transformer.transformer_blocks:
            for attn_name in ("attn1", "attn2"):
                attn = getattr(block, attn_name, None)
                if attn is None:
                    continue
                shard_specs[attn.to_q.weight] = ("model", None)
                shard_specs[attn.to_k.weight] = ("model", None)
                shard_specs[attn.to_v.weight] = ("model", None)
                shard_specs[attn.to_out[0].weight] = (None, "model")
            ff = getattr(block, "ff", None)
            if ff is not None and hasattr(ff, "net"):
                # net[0] is GEGLU/proj (column), net[-1] is the output (row)
                proj = getattr(ff.net[0], "proj", None)
                if proj is not None:
                    shard_specs[proj.weight] = ("model", None)
                shard_specs[ff.net[-1].weight] = (None, "model")
        return shard_specs

    @staticmethod
    def _shard_text_encoder(model):
        """Megatron-style GQA-TP map for the Gemma3 text encoder. KV projections
        are replicated (GQA fallback, same as the gemma4 bringup) — only
        q_proj/o_proj and the MLP gate/up/down are sharded."""
        shard_specs = {}
        te = getattr(model, "text_encoder", model)
        layers = te.model.language_model.layers
        for layer in layers:
            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", None)
            shard_specs[attn.o_proj.weight] = (None, "model")
            mlp = layer.mlp
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
        return shard_specs

    def load_shard_spec(self, model):
        """Megatron-style TP map. Non-sharded dim is ``None`` (replicated).

        Dispatches to the per-component sharding function. Only the weight-bound
        transformer / text encoder are sharded; the small components
        (connectors / vae / audio_vae / vocoder) replicate → empty map.
        """
        shard_fns = {
            ModelVariant.LTX2_TRANSFORMER: self._shard_transformer,
            ModelVariant.LTX2_TEXT_ENCODER: self._shard_text_encoder,
        }
        shard_fn = shard_fns.get(self._variant)
        return shard_fn(model) if shard_fn is not None else {}
