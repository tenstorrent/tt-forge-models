# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Coqui XTTS-v2 model loader implementation for multilingual text-to-speech.

XTTS-v2 (https://huggingface.co/coqui/XTTS-v2) is a multi-component TTS
pipeline, not a single forward pass. Its two compute-bearing neural networks
are brought up here as independent on-device components (mirroring the
composite-pipeline approach used for diffusion models):

  * ``gpt``              - the 30-layer, ~441M parameter autoregressive GPT-2
                          style transformer that maps text + speaker
                          conditioning to audio-code latents. This is the
                          compute-dominant core of XTTS.
  * ``hifigan_decoder``  - the ~26M parameter HiFi-GAN vocoder that turns the
                          GPT latents (plus a speaker embedding) into a
                          waveform.

The conditioning front-end (mel-spectrogram extraction + perceiver resampler +
speaker encoder) is deterministic preprocessing and is kept in host Python; its
fp32 outputs are fed into the on-device components as constant inputs. This is
the same "glue stays on host, heavy networks run on device" split that XTTS's
own ``inference()`` uses.

The GPT component is exercised through a small static-shape wrapper that
reproduces XTTS's ``GPT.forward(..., return_latent=True)`` latent computation
(text/audio token + positional embeddings -> GPT-2 transformer -> final norm ->
latent slice) without the data-dependent ``.max()``-based length bookkeeping of
the original forward, so the graph compiles with fully static shapes.
"""

from typing import Optional

import torch
import torch.nn.functional as F

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


def _apply_compat_shims():
    """Make coqui-tts importable on the modern transformers pinned by tt-xla.

    coqui-tts targets the transformers 4.x API. transformers 5.x removed
    ``isin_mps_friendly`` from ``transformers.pytorch_utils`` (imported by
    TTS's tortoise autoregressive module). Re-provide it as a thin alias for
    ``torch.isin`` before TTS is imported.
    """
    import transformers.pytorch_utils as ptu

    if not hasattr(ptu, "isin_mps_friendly"):
        ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(
            elements, test_elements
        )


class _NullPositionEmbeddings(torch.nn.Module):
    """Dtype-correct replacement for coqui's ``null_position_embeddings``.

    XTTS disables GPT-2's built-in positional embeddings (it uses external
    ``LearnedPositionEmbeddings``) by swapping ``GPT2Model.wpe`` for a partial
    that returns all-zeros. That partial hard-codes fp32, which silently
    upcasts the hidden states and breaks bf16 execution. This module returns
    the same zeros but in the module's (cast-tracked) dtype.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Cast by .to(dtype); used only to track the active floating dtype.
        self.register_buffer("_dtype_ref", torch.zeros(1), persistent=False)

    def forward(self, position_ids):
        return torch.zeros(
            *position_ids.shape,
            self.dim,
            dtype=self._dtype_ref.dtype,
            device=position_ids.device,
        )


class _GptLatentWrapper(torch.nn.Module):
    """Static-shape wrapper over XTTS's GPT producing audio-code latents.

    Equivalent to ``GPT.forward(..., return_latent=True)`` for fixed input
    lengths: text and audio token sequences (already start/stop padded on the
    host) are embedded, concatenated after the precomputed speaker conditioning
    latents, run through the GPT-2 transformer stack, normalized, and the mel
    portion is returned (dropping the trailing 5 positions, matching XTTS eval).
    """

    def __init__(self, gpt: torch.nn.Module):
        super().__init__()
        self.text_embedding = gpt.text_embedding
        self.text_pos_embedding = gpt.text_pos_embedding
        self.mel_embedding = gpt.mel_embedding
        self.mel_pos_embedding = gpt.mel_pos_embedding
        self.gpt = gpt.gpt  # transformers GPT2Model (30 layers)
        self.final_norm = gpt.final_norm
        # Replace the fp32-locked null position embedding so bf16 works.
        self.gpt.wpe = _NullPositionEmbeddings(self.gpt.config.n_embd)

    def forward(self, text_inputs, audio_codes, cond_latents):
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(
            text_inputs
        )
        mel_emb = self.mel_embedding(audio_codes) + self.mel_pos_embedding(audio_codes)
        emb = torch.cat([cond_latents, text_emb, mel_emb], dim=1)
        gpt_out = self.gpt(
            inputs_embeds=emb,
            return_dict=True,
            output_attentions=False,
            attention_mask=None,
        )
        enc = gpt_out.last_hidden_state[:, cond_latents.shape[1] :]
        enc = self.final_norm(enc)
        mel_latent = enc[:, -mel_emb.shape[1] :]
        return mel_latent[:, :-5]


class _HifiganDecoderWrapper(torch.nn.Module):
    """Wrapper over XTTS's HiFi-GAN decoder (latents + speaker emb -> waveform).

    Reimplements ``HifiDecoder.forward`` rather than calling it directly: the
    library version ends each interpolation with a ``.squeeze`` that is a no-op
    for the actual shapes (the squeezed dim is not size 1) and therefore traces
    to an aliasing ``prims::view_of``, which the TT functionalization pass
    rejects. Cloning after the interpolation removes the alias annotation while
    keeping the math identical.
    """

    def __init__(self, hifigan_decoder: torch.nn.Module):
        super().__init__()
        self.hifigan_decoder = hifigan_decoder

    def forward(self, latents, g):
        dec = self.hifigan_decoder
        z = F.interpolate(
            latents.transpose(1, 2),
            scale_factor=[dec.ar_mel_length_compression / dec.output_hop_length],
            mode="linear",
        ).clone()
        if dec.output_sample_rate != dec.input_sample_rate:
            z = F.interpolate(
                z,
                scale_factor=[dec.output_sample_rate / dec.input_sample_rate],
                mode="linear",
            ).clone()
        return dec.waveform_decoder(z, g=g)


class ModelVariant(StrEnum):
    """Available XTTS-v2 components to bring up."""

    GPT = "gpt"
    HIFIGAN_DECODER = "hifigan_decoder"


class ModelLoader(ForgeModel):
    """Coqui XTTS-v2 loader (per-component bring-up of the GPT and HiFi-GAN decoder)."""

    _VARIANTS = {
        ModelVariant.GPT: ModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
        ModelVariant.HIFIGAN_DECODER: ModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT

    # Reference conditioning is computed from a deterministic synthetic waveform
    # so bring-up needs no external audio file / audio codec.
    _REF_SECONDS = 6
    _REF_SR = 22050
    # Representative (dummy but in-range) text / audio-code sequence lengths.
    # The GPT's op coverage comes from its 30 transformer layers, not the
    # sequence length, so these are kept short to keep the runner's CPU golden
    # (run in bf16) well within the hardware-test timeout.
    _NUM_TEXT_TOKENS = 16
    _NUM_AUDIO_TOKENS = 24
    # Latent frames fed to the HiFi-GAN vocoder. Kept small because the vocoder
    # upsamples each frame to ~1k waveform samples and the bf16 CPU golden over
    # its conv stack is the slow path.
    _HIFIGAN_LATENT_FRAMES = 16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._full_model = None
        self._cond_latents = None  # fp32 (1, 32, 1024)
        self._speaker_embedding = None  # fp32 (1, 512, 1)
        self._gpt_latents = None  # fp32 (1, T, 1024) - input to the vocoder
        self._text_inputs = None  # long (1, T_text)
        self._audio_codes = None  # long (1, T_audio)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="xtts_v2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _synthetic_reference_waveform(self):
        """Deterministic 6s, 22.05kHz mono reference clip for speaker conditioning."""
        n = int(self._REF_SECONDS * self._REF_SR)
        t = torch.arange(n, dtype=torch.float32) / self._REF_SR
        two_pi = 2.0 * 3.141592653589793
        wav = 0.1 * torch.sin(two_pi * 220.0 * t) + 0.05 * torch.sin(two_pi * 440.0 * t)
        return wav.unsqueeze(0)  # (1, N)

    def _ensure_loaded(self):
        """Load the full fp32 XTTS model once and precompute fp32 conditioning.

        Done while the model is still fp32 (the mel/STFT conditioning front-end
        is numerically sensitive) and cached, so a later bf16 cast of the
        on-device wrapper does not affect these host-computed inputs.
        """
        if self._full_model is not None:
            return

        _apply_compat_shims()
        from huggingface_hub import snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        ckpt_dir = snapshot_download(
            self._variant_config.pretrained_model_name,
            allow_patterns=["*.json", "*.pth"],
        )
        config = XttsConfig()
        config.load_json(ckpt_dir + "/config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config, checkpoint_dir=ckpt_dir, use_deepspeed=False, eval=True
        )
        model.eval()
        self._full_model = model

        gpt = model.gpt
        wav = self._synthetic_reference_waveform()
        with torch.no_grad():
            # Conditioning front-end (host, fp32).
            self._cond_latents = model.get_gpt_cond_latents(wav, self._REF_SR)
            self._speaker_embedding = model.get_speaker_embedding(wav, self._REF_SR)

            # Deterministic, in-range text + audio-code sequences, start/stop
            # padded exactly as XTTS's GPT.forward does (host side, static).
            gen = torch.Generator().manual_seed(0)
            text_core = torch.randint(
                0,
                gpt.number_text_tokens - 3,
                (1, self._NUM_TEXT_TOKENS),
                generator=gen,
                dtype=torch.long,
            )
            audio_core = torch.randint(
                0,
                gpt.start_audio_token,
                (1, self._NUM_AUDIO_TOKENS),
                generator=gen,
                dtype=torch.long,
            )
            text_inputs = F.pad(text_core, (1, 0), value=gpt.start_text_token)
            text_inputs = F.pad(text_inputs, (0, 1), value=gpt.stop_text_token)
            audio_codes = F.pad(audio_core, (1, 0), value=gpt.start_audio_token)
            audio_codes = F.pad(audio_codes, (0, 1), value=gpt.stop_audio_token)
            self._text_inputs = text_inputs
            self._audio_codes = audio_codes

            # GPT latents that feed the vocoder (host, fp32) - the real network
            # output, so the HiFi-GAN component is exercised on faithful inputs.
            gpt_wrapper = _GptLatentWrapper(gpt)
            gpt_wrapper.eval()
            self._gpt_latents = gpt_wrapper(text_inputs, audio_codes, self._cond_latents)

    def load_model(self, dtype_override=None):
        """Return the on-device component (nn.Module) for this variant.

        Args:
            dtype_override: Optional torch.dtype applied to the component
                weights (the runner passes torch.bfloat16).
        """
        self._ensure_loaded()

        if self._variant == ModelVariant.HIFIGAN_DECODER:
            model = _HifiganDecoderWrapper(self._full_model.hifigan_decoder)
            # The HiFi-GAN vocoder's deep weight-normed conv stack is highly
            # precision sensitive: bf16 drops device-vs-CPU PCC to ~0.32, so it
            # is brought up in fp32 regardless of any requested override.
            dtype_override = None
        else:
            model = _GptLatentWrapper(self._full_model.gpt)

        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        """Return the sample inputs dict for this variant's forward pass.

        Args:
            dtype_override: Optional torch.dtype applied to floating-point
                inputs (integer token tensors are left untouched).
        """
        self._ensure_loaded()

        if self._variant == ModelVariant.HIFIGAN_DECODER:
            # Keep the vocoder inputs in fp32 to match its fp32 weights (see
            # load_model); bf16 is numerically inadequate for this component.
            latents = self._gpt_latents[:, : self._HIFIGAN_LATENT_FRAMES].clone()
            g = self._speaker_embedding.clone()
            return {"latents": latents, "g": g}

        cond_latents = self._cond_latents.clone()
        if dtype_override is not None:
            cond_latents = cond_latents.to(dtype_override)
        return {
            "text_inputs": self._text_inputs.clone(),
            "audio_codes": self._audio_codes.clone(),
            "cond_latents": cond_latents,
        }
