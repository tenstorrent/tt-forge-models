# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Coqui XTTS-v2 text-to-speech model loader implementation.

Bringup target = the autoregressive GPT-2 transformer core of XTTS-v2
(``GPT2Model`` 378.9M params) + final LayerNorm + mel logits head. This is the
compute-dominant single forward pass. The HiFi-GAN vocoder and the
data-dependent autoregressive sampling loop are out of scope for device bringup
(unsupported vocoder ops / dynamic output length), mirroring how other TTS
backbones are brought up as their transformer core only.

The forward is driven with precomputed ``inputs_embeds`` =
cat([speaker conditioning latent, text embeddings, mel embeddings]), exactly the
sequence XTTS' ``GPT.get_logits`` feeds to the GPT-2 core. The speaker
conditioning latent ships precomputed per-speaker in ``speakers_xtts.pth``, so
the inputs are fully reproducible without any reference audio file.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# A fixed, reproducible prompt and mel length for the sample forward pass.
_SAMPLE_TEXT = "It took me quite a long time to develop a voice."
_SAMPLE_LANG = "en"
_MEL_LEN = 32


def _apply_transformers_shim():
    """coqui-tts 0.27.x imports ``isin_mps_friendly`` from
    ``transformers.pytorch_utils``, which was removed in transformers>=5. Shim it
    before importing TTS so the package imports against transformers 5.x without
    downgrading the stack (the device-test harness depends on transformers 5.5.1).
    """
    import transformers.pytorch_utils as ptu

    if not hasattr(ptu, "isin_mps_friendly"):
        ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(
            elements, test_elements
        )


class _NullPositionEmbeddings(torch.nn.Module):
    """XTTS zeroes GPT-2's learned positional embeddings (positional information is
    injected separately via the text/mel positional embeddings). The stock
    replacement returns *float32* zeros, which promotes bf16 hidden states and
    triggers a mixed-dtype error in the downstream LayerNorm on device. This
    cast-aware variant returns zeros in the module's own dtype, so ``.to(bf16)``
    keeps the whole GPT-2 stack in a single dtype.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Buffer exists only to track the module's dtype across .to() casts.
        self.register_buffer("_dtype_probe", torch.zeros(1), persistent=False)

    def forward(self, position_ids):
        return torch.zeros(
            position_ids.shape[0],
            position_ids.shape[1],
            self.dim,
            device=position_ids.device,
            dtype=self._dtype_probe.dtype,
        )


class XttsGptCore(torch.nn.Module):
    """The bringup target: XTTS-v2's GPT-2 transformer core + final norm + mel head.

    forward(inputs_embeds) -> mel logits over the precomputed embedding sequence.
    """

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt.gpt  # transformers GPT2Model (378.9M params)
        self.final_norm = gpt.final_norm
        self.mel_head = gpt.mel_head
        # Replace the float32-zeros positional-embedding stub with a dtype-aware one.
        self.gpt.wpe = _NullPositionEmbeddings(gpt.model_dim)

    def forward(self, inputs_embeds):
        gpt_out = self.gpt(inputs_embeds=inputs_embeds, return_dict=True)
        enc = self.final_norm(gpt_out.last_hidden_state)
        return self.mel_head(enc)


class ModelVariant(StrEnum):
    """Available XTTS-v2 model variants."""

    V2 = "v2"


class ModelLoader(ForgeModel):
    """Coqui XTTS-v2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.V2: LLMModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="xtts_v2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._xtts = None
        self._local_dir = None

    def _load_xtts(self):
        """Build and cache the full Coqui Xtts model (used for both the GPT core
        wrapper and to compute the embedding inputs)."""
        if self._xtts is not None:
            return self._xtts

        _apply_transformers_shim()
        from huggingface_hub import snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        self._local_dir = snapshot_download(
            self._variant_config.pretrained_model_name
        )
        config = XttsConfig()
        config.load_json(os.path.join(self._local_dir, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=self._local_dir,
            use_deepspeed=False,
            eval=True,
        )
        model.eval()
        self._xtts = model
        return self._xtts

    def load_model(self, dtype_override=None):
        """Load the XTTS-v2 GPT transformer core (single forward pass target)."""
        xtts = self._load_xtts()
        core = XttsGptCore(xtts.gpt).eval()
        if dtype_override is not None:
            core = core.to(dtype_override)
        return core

    def load_inputs(self, dtype_override=None):
        """Build the precomputed GPT-core input embeddings.

        Returns ``inputs_embeds`` = cat([speaker conditioning latent, text
        embeddings, mel embeddings]) — the exact sequence XTTS feeds to its
        GPT-2 core. Reproducible from precomputed speaker latents (no audio file).
        """
        xtts = self._load_xtts()
        g = xtts.gpt

        # Speaker conditioning latent: precomputed (perceiver-resampled) per speaker.
        speakers = torch.load(
            os.path.join(self._local_dir, "speakers_xtts.pth"),
            map_location="cpu",
            weights_only=False,
        )
        cond_latent = next(iter(speakers.values()))["gpt_cond_latent"]  # [1, 32, 1024]

        # Text branch: encode the fixed prompt, prepend start / append stop tokens
        # exactly as GPT.forward does, then text embedding + positional embedding.
        text_ids = xtts.tokenizer.encode(_SAMPLE_TEXT, lang=_SAMPLE_LANG)
        text_inputs = torch.tensor([text_ids], dtype=torch.long)
        text_inputs = F.pad(text_inputs, (0, 1), value=g.stop_text_token)
        text_inputs, _ = g.set_inputs_and_targets(
            text_inputs, g.start_text_token, g.stop_text_token
        )
        text_emb = g.text_embedding(text_inputs) + g.text_pos_embedding(text_inputs)

        # Mel branch: a fixed, deterministic mel-code sequence (real codes come from
        # the DVAE during inference; content is irrelevant for the forward-pass
        # bringup as long as CPU and device see identical inputs).
        mel_codes = (
            torch.arange(_MEL_LEN, dtype=torch.long) % g.num_audio_tokens
        ).unsqueeze(0)
        mel_codes = F.pad(mel_codes, (0, 1), value=g.stop_audio_token)
        mel_codes, _ = g.set_inputs_and_targets(
            mel_codes, g.start_audio_token, g.stop_audio_token
        )
        mel_emb = g.mel_embedding(mel_codes) + g.mel_pos_embedding(mel_codes)

        inputs_embeds = torch.cat([cond_latent, text_emb, mel_emb], dim=1)
        if dtype_override is not None:
            inputs_embeds = inputs_embeds.to(dtype_override)
        return {"inputs_embeds": inputs_embeds}
