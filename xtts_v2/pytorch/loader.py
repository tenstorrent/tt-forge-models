# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XTTS-v2 model loader implementation for text-to-speech tasks.

coqui/XTTS-v2 is a Coqui TTS pipeline. Its compute-dominant component is an
autoregressive GPT-2 backbone ("the GPT core") that consumes a sequence of
conditioning latents + text-token embeddings + mel-token embeddings and predicts
the next mel (audio-codebook) token. The downstream HiFi-GAN decoder that turns
mel codes into a waveform is out of scope for this single-forward bringup.

The official checkpoint (model.pth) is not a `transformers` model and the upstream
`TTS` library pins an old torch/transformers stack that would break the tt-xla
device path, so this loader reconstructs the GPT core directly from
`transformers.GPT2Model` plus the checkpoint's external embedding/head layers,
loading the `gpt.*` weights straight from model.pth. The forward pass is a single
static-shape pass over `[cond_latents, text_emb, mel_emb]` producing mel-token
logits -- matching how the GPT core is exercised inside the XTTS inference loop.
"""
import pickle
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import GPT2Config, GPT2Model

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


# ---------------------------------------------------------------------------
# GPT-core architecture (reconstructed from the checkpoint's `gpt.*` weights).
# Hyper-parameters come from coqui/XTTS-v2 config.json -> model_args.
# ---------------------------------------------------------------------------
GPT_N_LAYERS = 30
GPT_N_HEADS = 16
GPT_N_MODEL_CHANNELS = 1024
GPT_N_INNER = 4096  # GPT-2 mlp.c_fc -> 4 * model_dim
GPT_NUMBER_TEXT_TOKENS = 6681
GPT_NUM_AUDIO_TOKENS = 1026
GPT_MAX_TEXT_TOKENS = 402  # text_pos_embedding capacity is 404
GPT_MAX_AUDIO_TOKENS = 605  # mel_pos_embedding capacity is 608
GPT_START_AUDIO_TOKEN = 1024
GPT_NUM_COND_LATENTS = 32  # perceiver-resampler output length


class LearnedPositionEmbeddings(nn.Module):
    """Matches coqui's LearnedPositionEmbeddings (key: `<name>.emb.weight`)."""

    def __init__(self, seq_len: int, model_dim: int):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)

    def forward(self, seq_len: int, device) -> torch.Tensor:
        return self.emb(torch.arange(0, seq_len, device=device))


class XttsGptCore(nn.Module):
    """The XTTS-v2 autoregressive GPT-2 backbone, single static forward.

    Inputs:
        cond_latents: (B, 32, 1024) speaker-conditioning latents (a precomputed
            input -- in the full pipeline these come from the perceiver resampler
            over the reference audio; the conditioning front-end and the HiFi-GAN
            vocoder are out of scope for this forward-pass bringup).
        text_ids:     (B, T_text) text token ids in [0, 6681).
        mel_ids:      (B, T_mel)  mel (audio-codebook) token ids in [0, 1026).

    Output:
        mel_logits:   (B, T_mel, 1026) next-mel-token logits.
    """

    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Embedding(GPT_NUMBER_TEXT_TOKENS, GPT_N_MODEL_CHANNELS)
        self.mel_embedding = nn.Embedding(GPT_NUM_AUDIO_TOKENS, GPT_N_MODEL_CHANNELS)
        self.text_pos_embedding = LearnedPositionEmbeddings(
            GPT_MAX_TEXT_TOKENS + 2, GPT_N_MODEL_CHANNELS
        )
        self.mel_pos_embedding = LearnedPositionEmbeddings(
            GPT_MAX_AUDIO_TOKENS + 3, GPT_N_MODEL_CHANNELS
        )

        gpt_config = GPT2Config(
            vocab_size=256,  # dummy; wte is unused (we feed inputs_embeds)
            n_positions=GPT_MAX_AUDIO_TOKENS + GPT_MAX_TEXT_TOKENS + GPT_NUM_COND_LATENTS + 8,
            n_embd=GPT_N_MODEL_CHANNELS,
            n_layer=GPT_N_LAYERS,
            n_head=GPT_N_HEADS,
            n_inner=GPT_N_INNER,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.gpt = GPT2Model(gpt_config)
        # coqui replaces the GPT-2 positional embedding with zeros (positions are
        # handled by the external text/mel pos embeddings above).
        with torch.no_grad():
            self.gpt.wpe.weight.zero_()

        self.final_norm = nn.LayerNorm(GPT_N_MODEL_CHANNELS)
        self.text_head = nn.Linear(GPT_N_MODEL_CHANNELS, GPT_NUMBER_TEXT_TOKENS)
        self.mel_head = nn.Linear(GPT_N_MODEL_CHANNELS, GPT_NUM_AUDIO_TOKENS)

    def forward(self, cond_latents, text_ids, mel_ids):
        device = text_ids.device
        text_emb = self.text_embedding(text_ids) + self.text_pos_embedding(
            text_ids.shape[1], device
        )
        mel_emb = self.mel_embedding(mel_ids) + self.mel_pos_embedding(
            mel_ids.shape[1], device
        )
        cond_latents = cond_latents.to(text_emb.dtype)

        emb = torch.cat([cond_latents, text_emb, mel_emb], dim=1)
        # Pass an explicit all-ones (no-padding) attention mask. Besides being the
        # faithful input, this keeps transformers' unified masking off the
        # `find_packed_sequence_indices` branch, whose bool `cumsum` lowers to an
        # unsupported UInt8 tt-metal accumulation kernel. The mask is int32 so the
        # remaining (left-padding) cumsum stays on a supported integer format.
        attention_mask = torch.ones(
            emb.shape[0], emb.shape[1], dtype=torch.int32, device=emb.device
        )
        hidden = self.gpt(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = hidden.last_hidden_state

        # Drop the conditioning region; keep [text | mel]; normalize; mel head.
        text_len = text_ids.shape[1]
        enc = self.final_norm(hidden[:, GPT_NUM_COND_LATENTS:])
        mel_logits = self.mel_head(enc[:, text_len:])
        return mel_logits


# ---------------------------------------------------------------------------
# Checkpoint loading helpers.
# ---------------------------------------------------------------------------
class _StubObject:
    """Placeholder for any unpickled `TTS.*` object (e.g. the bundled config)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickle coqui/XTTS-v2/model.pth without the (heavy, version-pinned) `TTS`
    package: the checkpoint pickles a `TTS.*` config object alongside the weights,
    so map every `TTS.*` class to a harmless stub. Leaves `sys.modules` untouched,
    so nothing leaks into the torch.compile tracing path later."""

    def find_class(self, module, name):
        if module == "TTS" or module.startswith("TTS."):
            return _StubObject
        return super().find_class(module, name)


class _TolerantPickleModule:
    """`pickle_module` shim exposing the tolerant Unpickler to torch.load."""

    Unpickler = _TolerantUnpickler
    Pickler = pickle.Pickler
    load = staticmethod(pickle.load)
    dump = staticmethod(pickle.dump)


class ModelVariant(StrEnum):
    """Available XTTS-v2 model variants."""

    XTTS_V2 = "xtts_v2"


class ModelLoader(ForgeModel):
    """XTTS-v2 GPT-core loader for the text-to-speech (mel-token generation) task."""

    _VARIANTS = {
        ModelVariant.XTTS_V2: ModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XTTS_V2

    # Static input sizes for the single-forward bringup.
    TEXT_SEQ_LEN = 32
    MEL_SEQ_LEN = 32

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._ckpt = None
        self._cond_prior = None

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

    def _load_checkpoint(self):
        """Download model.pth and return the `gpt.*` sub-state-dict (cached)."""
        if self._ckpt is not None:
            return self._ckpt
        path = hf_hub_download(
            self._variant_config.pretrained_model_name, "model.pth"
        )
        full = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_TolerantPickleModule,
        )
        state = full["model"] if isinstance(full, dict) and "model" in full else full
        gpt_sd = {
            k[len("gpt.") :]: v for k, v in state.items() if k.startswith("gpt.")
        }
        # Perceiver-resampler seed latents used as the speaker-conditioning prior.
        self._cond_prior = gpt_sd["conditioning_perceiver.latents"].clone()
        self._ckpt = gpt_sd
        return gpt_sd

    def load_model(self, dtype_override=None):
        gpt_sd = self._load_checkpoint()
        model = XttsGptCore()

        # Only load the layers we reconstruct (text/mel embeddings + pos
        # embeddings + GPT-2 stack + final_norm + heads). The conditioning
        # encoder / perceiver and the HiFi-GAN decoder are out of scope.
        wanted = dict(model.state_dict())
        to_load = {k: v for k, v in gpt_sd.items() if k in wanted}
        missing, unexpected = model.load_state_dict(to_load, strict=False)
        # `missing` should only be the GPT-2 internal wte/wpe (not in checkpoint;
        # wpe is intentionally zeroed, wte is unused with inputs_embeds).
        leftover = [m for m in missing if not m.startswith("gpt.wte") and not m.startswith("gpt.wpe")]
        assert not leftover, f"Unexpected missing GPT-core weights: {leftover}"

        model = model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        if self._cond_prior is None:
            self._load_checkpoint()

        batch_size = 1
        text_len = self.TEXT_SEQ_LEN
        mel_len = self.MEL_SEQ_LEN

        # Deterministic, in-range token ids (reproducible CPU<->TT comparison).
        text_ids = (torch.arange(text_len) % GPT_NUMBER_TEXT_TOKENS).unsqueeze(0)
        text_ids = text_ids.repeat(batch_size, 1).to(torch.int64)

        mel_ids = (torch.arange(mel_len) % (GPT_START_AUDIO_TOKEN)).unsqueeze(0)
        mel_ids[:, 0] = GPT_START_AUDIO_TOKEN  # start-of-audio token
        mel_ids = mel_ids.repeat(batch_size, 1).to(torch.int64)

        # Speaker-conditioning latents: the perceiver's learned seed latents used
        # as a fixed conditioning prior (the audio-driven conditioning front-end
        # is out of scope for this single-forward bringup).
        cond_latents = self._cond_prior.unsqueeze(0).repeat(batch_size, 1, 1).float()
        if dtype_override is not None:
            cond_latents = cond_latents.to(dtype_override)

        return {
            "cond_latents": cond_latents,
            "text_ids": text_ids,
            "mel_ids": mel_ids,
        }
