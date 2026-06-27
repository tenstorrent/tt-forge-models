# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XTTS-v2 model loader implementation.

XTTS-v2 (coqui/XTTS-v2) is a multilingual zero-shot voice-cloning TTS pipeline
built around a 441M-parameter autoregressive GPT (a GPT-2 transformer core that
predicts discrete mel/audio tokens) plus a HiFi-GAN vocoder and a DVAE. The
compute-dominant, transformer part is the GPT core (a GPT2Model, 378.9M params).

This loader brings up exactly that GPT transformer core (gpt.gpt + final_norm +
mel_head) as a single static forward pass. The autoregressive sampling loop and
the HiFi-GAN vocoder are intentionally out of scope: they involve
data-dependent sequence lengths and vocoder ops that are not part of a single
forward graph. We drive the core with a precomputed ``inputs_embeds`` tensor
``cat([cond_latents, text_emb, mel_emb])`` exactly as XTTS assembles it in
``GPT.get_logits``, so the bringup is fully reproducible from a precomputed
per-speaker conditioning latent (shipped in ``speakers_xtts.pth``) and a fixed
text/mel token sequence — no reference audio file is needed.

Notes (see the family requirements.txt):
- coqui-tts (the maintained idiap fork) still imports ``isin_mps_friendly`` from
  ``transformers.pytorch_utils``, which was removed in transformers>=5. We shim
  it before importing TTS rather than downgrading transformers.
- XTTS replaces the GPT-2 ``wpe`` with a function that returns float32 zeros;
  under a bf16 run that promotes the hidden states and triggers a mixed-dtype
  LayerNorm error. We swap it for a dtype-aware null module instead.
"""
import torch
import torch.nn as nn
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


def _install_transformers5_shim():
    """coqui-tts 0.27.x imports ``isin_mps_friendly`` (removed in transformers>=5).

    Provide it on ``transformers.pytorch_utils`` before TTS is imported, instead
    of downgrading transformers (the device-test harness depends on 5.x).
    """
    import transformers.pytorch_utils as ptu

    if not hasattr(ptu, "isin_mps_friendly"):
        ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(
            elements, test_elements
        )


class _NullPositionEmbeddings(nn.Module):
    """Dtype-aware replacement for XTTS' ``null_position_embeddings``.

    XTTS disables GPT-2's learned position embeddings by setting ``wpe`` to a
    function returning ``torch.zeros(...)`` in the default (float32) dtype. When
    the model is cast to bf16 that float32 output promotes the bf16 hidden
    states and the following LayerNorm fails with a mixed-dtype error. This
    module returns zeros that track the model's dtype: the registered buffer is
    cast by ``module.to(dtype)`` along with the rest of the weights.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Persistent=False: this is not a real parameter, just a dtype carrier.
        self.register_buffer("_dtype_ref", torch.zeros(1), persistent=False)

    def forward(self, position_ids):
        return torch.zeros(
            position_ids.shape[0],
            position_ids.shape[1],
            self.dim,
            dtype=self._dtype_ref.dtype,
            device=position_ids.device,
        )


class XttsGptCore(nn.Module):
    """The XTTS-v2 autoregressive GPT transformer core as a single forward pass.

    Wraps the GPT-2 transformer (``gpt.gpt``), the ``final_norm`` LayerNorm and
    the ``mel_head`` projection. Given a precomputed ``inputs_embeds`` =
    ``cat([cond_latents, text_emb, mel_emb])`` it returns the mel-token logits
    for the mel region (shape ``[B, mel_len, num_audio_tokens]``), mirroring
    ``GPT.get_logits``.
    """

    def __init__(self, gpt2_model, final_norm, mel_head, mel_offset=0):
        super().__init__()
        self.gpt = gpt2_model
        self.final_norm = final_norm
        self.mel_head = mel_head
        # Number of leading (cond_latent + text) positions to drop before the
        # mel head; set from the fixed input shapes by the loader.
        self.mel_offset = mel_offset

    def forward(self, inputs_embeds):
        gpt_out = self.gpt(inputs_embeds=inputs_embeds, return_dict=True)
        enc = gpt_out.last_hidden_state[:, self.mel_offset :]
        enc = self.final_norm(enc)
        logits = self.mel_head(enc)
        return logits


class ModelVariant(StrEnum):
    """Available XTTS-v2 model variants."""

    V2 = "v2"


class ModelLoader(ForgeModel):
    """XTTS-v2 GPT-core loader implementation for text-to-speech."""

    _VARIANTS = {
        ModelVariant.V2: ModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2

    # Fixed demo text and speaker, kept constant so the assembled inputs are
    # fully reproducible across CPU and device runs.
    DEFAULT_TEXT = (
        "It took me quite a long time to develop a voice, and now that I "
        "have it I am not going to be silent."
    )
    DEFAULT_SPEAKER = "Claribel Dervla"
    # Number of mel tokens to feed (after the start_audio_token). Small and
    # fixed: this is a single forward pass, not the autoregressive loop.
    NUM_MEL_TOKENS = 32

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._xtts = None
        self._gpt = None
        self._model = None

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

    def _load_xtts(self):
        """Load the full XTTS-v2 pipeline (cached on the instance)."""
        if self._xtts is not None:
            return self._xtts

        _install_transformers5_shim()
        import os
        from huggingface_hub import hf_hub_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        repo = self._variant_config.pretrained_model_name
        cfg_path = hf_hub_download(repo, "config.json")
        ckpt_dir = os.path.dirname(cfg_path)

        config = XttsConfig()
        config.load_json(cfg_path)
        xtts = Xtts.init_from_config(config)
        xtts.load_checkpoint(
            config, checkpoint_dir=ckpt_dir, use_deepspeed=False, eval=True
        )
        xtts.eval()
        self._xtts = xtts
        self._gpt = xtts.gpt
        return xtts

    def load_model(self, dtype_override=None):
        """Load the XTTS-v2 GPT transformer core as a single-forward module.

        Returns:
            XttsGptCore: gpt.gpt (GPT-2 core, with dtype-aware null position
            embeddings) + final_norm + mel_head.
        """
        xtts = self._load_xtts()
        gpt = self._gpt

        # Replace the float32-zeros position-embedding function with a
        # dtype-aware module so a bf16 run does not hit a mixed-dtype LayerNorm.
        gpt.gpt.wpe = _NullPositionEmbeddings(gpt.gpt.config.n_embd)

        model = XttsGptCore(gpt.gpt, gpt.final_norm, gpt.mel_head)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._model = model
        return model

    def _build_inputs_embeds(self, dtype_override=None):
        """Assemble inputs_embeds = cat([cond_latents, text_emb, mel_emb]).

        Mirrors XTTS' GPT.get_logits / compute_embeddings assembly. Also records
        the mel offset (cond + text length) on the wrapped model so its forward
        slices the mel region.
        """
        xtts = self._load_xtts()
        gpt = self._gpt

        # 1. Per-speaker conditioning latent, shipped precomputed in
        #    speakers_xtts.pth (no reference audio needed).
        from huggingface_hub import hf_hub_download

        spk_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "speakers_xtts.pth"
        )
        speakers = torch.load(spk_path, map_location="cpu", weights_only=False)
        cond_latents = speakers[self.DEFAULT_SPEAKER]["gpt_cond_latent"].float()  # [1,32,1024]

        # 2. Text tokens -> embeddings + learned positions, with the same
        #    start/stop padding XTTS applies in compute_embeddings.
        text_tokens = torch.tensor(
            xtts.tokenizer.encode(self.DEFAULT_TEXT, lang="en"), dtype=torch.long
        ).unsqueeze(0)
        text_tokens = torch.nn.functional.pad(
            text_tokens, (0, 1), value=gpt.stop_text_token
        )
        text_tokens = torch.nn.functional.pad(
            text_tokens, (1, 0), value=gpt.start_text_token
        )
        text_emb = gpt.text_embedding(text_tokens) + gpt.text_pos_embedding(text_tokens)

        # 3. Mel tokens: start_audio_token followed by a fixed deterministic
        #    sequence (this is a single forward pass, not the AR sampling loop).
        mel_tokens = torch.arange(self.NUM_MEL_TOKENS, dtype=torch.long) % gpt.num_audio_tokens
        mel_tokens = torch.cat(
            [torch.tensor([gpt.start_audio_token], dtype=torch.long), mel_tokens]
        ).unsqueeze(0)
        mel_emb = gpt.mel_embedding(mel_tokens) + gpt.mel_pos_embedding(mel_tokens)

        inputs_embeds = torch.cat([cond_latents, text_emb, mel_emb], dim=1)

        # mel region begins after cond_latents + text positions.
        mel_offset = cond_latents.shape[1] + text_emb.shape[1]
        if self._model is not None:
            self._model.mel_offset = mel_offset

        if dtype_override is not None:
            inputs_embeds = inputs_embeds.to(dtype_override)

        return inputs_embeds

    def load_inputs(self, dtype_override=None):
        """Return the precomputed inputs for the XTTS-v2 GPT core."""
        inputs_embeds = self._build_inputs_embeds(dtype_override=dtype_override)
        return {"inputs_embeds": inputs_embeds}
