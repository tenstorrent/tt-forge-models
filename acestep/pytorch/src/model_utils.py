# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loading / input helpers for ACE-Step 1.5 (text-to-music).

ACE-Step 1.5 is a multi-stage generative music pipeline, brought up by
independently-compilable components (see the diffusion bringup convention):

  - TextEncoder  -> Qwen3-Embedding-0.6B  (text-prompt conditioning encoder)
  - Lm           -> acestep-5Hz-lm-1.7B   (Qwen3 semantic-token language model)
  - Denoiser     -> AceStepDiTModel       (the per-step flow-matching DiT; key component)
  - VaeDecoder   -> AutoencoderOobleck     (48 kHz stereo audio decoder)

The turbo checkpoint ships as `trust_remote_code` custom modeling. Loading the
full `AceStepConditionGenerationModel` via `from_pretrained` forces meta-device
init, which trips an `.item()` assert inside the FSQ audio tokenizer's __init__.
The denoiser only needs the DiT sub-module, so we import the vendored modeling
file directly, instantiate `AceStepDiTModel(config)` on CPU, and load just the
`decoder.*` weights from the checkpoint. This sidesteps the FSQ tokenizer
entirely (it is used only for cover-song LM hints, not the denoise step).
"""

import importlib
import os
import sys

import torch
import torch.nn as nn

REPO_ID = "ACE-Step/Ace-Step1.5"
DTYPE = torch.bfloat16  # native distribution format of every component

# --- Native "resolution" for the artifact ---------------------------------
# Music duration is a free parameter (the DiT/VAE handle variable length). We
# fix a representative native clip of 30 s. Both DiT and the Oobleck VAE run at
# a 25 Hz, 64-channel latent, so 30 s == 750 latent frames.
LATENT_HZ = 25
NATIVE_SECONDS = 30
NATIVE_LATENT_FRAMES = NATIVE_SECONDS * LATENT_HZ  # 750

# Synthetic conditioning length fed to the denoiser (packed text+lyric+timbre).
COND_SEQ_LEN = 320

# Token sequence length for the text encoder / LM component checks.
TEXT_SEQ_LEN = 256

VAE_LATENT_CHANNELS = 64      # Oobleck decoder_input_channels
DIT_ACOUSTIC_DIM = 64         # audio_acoustic_hidden_dim (DiT latent width)
DIT_CONTEXT_DIM = 128         # src_latents(64) + chunk_masks(64)


# --------------------------------------------------------------------------
# Local snapshot helpers (custom code must sit next to config.json to import)
# --------------------------------------------------------------------------
def _component_dir(subfolder: str) -> str:
    from huggingface_hub import snapshot_download

    root = snapshot_download(REPO_ID, allow_patterns=[f"{subfolder}/*"])
    return os.path.join(root, subfolder)


def _load_turbo_modeling(turbo_dir: str):
    """Import the vendored ACE-Step modeling + config modules from the snapshot."""
    if turbo_dir not in sys.path:
        sys.path.insert(0, turbo_dir)
    cfg_mod = importlib.import_module("configuration_acestep_v15")
    model_mod = importlib.import_module("modeling_acestep_v15_turbo")
    return cfg_mod, model_mod


# --------------------------------------------------------------------------
# Denoiser (key component): AceStepDiTModel
# --------------------------------------------------------------------------
class DenoiserWrapper(nn.Module):
    """Single flow-matching denoise step; returns the predicted velocity tensor.

    Masks are built internally by the DiT in eager mode (the incoming padding
    masks are ignored there), so we only need to forward the real inputs and
    disable the KV cache for a clean single-step graph.
    """

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit

    def forward(self, hidden_states, timestep, encoder_hidden_states, context_latents):
        bsz, seq = hidden_states.shape[0], hidden_states.shape[1]
        attention_mask = torch.ones(
            bsz, seq, dtype=hidden_states.dtype, device=hidden_states.device
        )
        enc_mask = torch.ones(
            bsz,
            encoder_hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        out = self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=enc_mask,
            context_latents=context_latents,
            use_cache=False,
        )
        return out[0]


def load_denoiser(dtype: torch.dtype) -> nn.Module:
    from safetensors.torch import load_file

    turbo_dir = _component_dir("acestep-v15-turbo")
    cfg_mod, model_mod = _load_turbo_modeling(turbo_dir)
    config = cfg_mod.AceStepConfig.from_pretrained(turbo_dir)
    config._attn_implementation = "eager"

    dit = model_mod.AceStepDiTModel(config)
    full_sd = load_file(os.path.join(turbo_dir, "model.safetensors"))
    dec_sd = {
        k[len("decoder.") :]: v
        for k, v in full_sd.items()
        if k.startswith("decoder.")
    }
    missing, unexpected = dit.load_state_dict(dec_sd, strict=False)
    assert not missing and not unexpected, (
        f"decoder weight mismatch: missing={missing[:4]} unexpected={unexpected[:4]}"
    )
    return DenoiserWrapper(dit.to(dtype).eval())


def make_denoiser_inputs(dtype: torch.dtype):
    b, t, tc = 1, NATIVE_LATENT_FRAMES, COND_SEQ_LEN
    hidden_states = torch.randn(b, t, DIT_ACOUSTIC_DIM, dtype=dtype)
    context_latents = torch.randn(b, t, DIT_CONTEXT_DIM, dtype=dtype)
    encoder_hidden_states = torch.randn(b, tc, 2048, dtype=dtype)
    timestep = torch.ones(b, dtype=dtype)  # t=1.0 -> first (highest-noise) step
    return [hidden_states, timestep, encoder_hidden_states, context_latents]


# --------------------------------------------------------------------------
# VAE decoder: AutoencoderOobleck (48 kHz stereo)
# --------------------------------------------------------------------------
class VaeDecoderWrapper(nn.Module):
    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        out = self.vae.decode(latents)
        return out.sample if hasattr(out, "sample") else out


def load_vae_decoder(dtype: torch.dtype) -> nn.Module:
    from diffusers import AutoencoderOobleck

    vae_dir = _component_dir("vae")
    vae = AutoencoderOobleck.from_pretrained(vae_dir).to(dtype).eval()
    return VaeDecoderWrapper(vae)


def make_vae_inputs(dtype: torch.dtype):
    return [torch.randn(1, VAE_LATENT_CHANNELS, NATIVE_LATENT_FRAMES, dtype=dtype)]


# --------------------------------------------------------------------------
# Text encoder (Qwen3-Embedding-0.6B) and LM (acestep-5Hz-lm-1.7B)
# --------------------------------------------------------------------------
class HiddenStateWrapper(nn.Module):
    """Return last_hidden_state so the comparison harness sees a single tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


def _load_qwen3(subfolder: str, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoModel

    comp_dir = _component_dir(subfolder)
    model = AutoModel.from_pretrained(
        comp_dir, attn_implementation="eager", dtype=dtype
    ).eval()
    return HiddenStateWrapper(model)


def load_text_encoder(dtype: torch.dtype) -> nn.Module:
    return _load_qwen3("Qwen3-Embedding-0.6B", dtype)


def load_lm(dtype: torch.dtype) -> nn.Module:
    return _load_qwen3("acestep-5Hz-lm-1.7B", dtype)


def _vocab_size(subfolder: str) -> int:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(_component_dir(subfolder)).vocab_size


def make_text_encoder_inputs():
    vocab = _vocab_size("Qwen3-Embedding-0.6B")
    ids = torch.randint(0, vocab, (1, TEXT_SEQ_LEN), dtype=torch.long)
    mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)
    return [ids, mask]


def make_lm_inputs():
    vocab = _vocab_size("acestep-5Hz-lm-1.7B")
    ids = torch.randint(0, vocab, (1, TEXT_SEQ_LEN), dtype=torch.long)
    mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)
    return [ids, mask]
