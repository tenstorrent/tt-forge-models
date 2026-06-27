# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XTTS-v2 model loader implementation for text-to-speech tasks.

coqui/XTTS-v2 is a coqui-library (not transformers) TTS model. Its compute is
dominated by an autoregressive GPT-2 transformer ("GPT") that, conditioned on a
speaker latent + text tokens, predicts discrete mel-code logits; a separate
HiFi-GAN decoder then turns the generated codes into a waveform.

This loader brings up the GPT transformer core as a single static forward pass:
the 379M-parameter GPT2Model backbone plus the final LayerNorm and mel head.
Conditioning / text / mel embeddings are precomputed on host and fed in as
``inputs_embeds``; the device graph runs the transformer stack and emits
mel-token logits. The HiFi-GAN vocoder and the autoregressive generation loop
are intentionally out of scope -- both rely on data-dependent output length and
vocoder ops that the static-shape device path does not support (see the TTS /
vocoder triage in the bringup references).
"""
import os
from typing import Optional

import torch
import torch.nn as nn

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


def _install_transformers_compat_shim():
    """coqui-tts 0.27.x imports ``isin_mps_friendly`` from
    ``transformers.pytorch_utils``, which was removed in transformers>=5. Provide
    a small shim so the TTS package imports under the pinned transformers 5.x.
    """
    import transformers.pytorch_utils as ptu

    if not hasattr(ptu, "isin_mps_friendly"):
        ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(
            elements, test_elements
        )


class _NullPositionEmbeddings(nn.Module):
    """Drop-in for the GPT2 positional embedding that XTTS zeroes out.

    XTTS replaces the GPT2 ``wpe`` with a function returning float32 zeros (it
    supplies its own learned positional embeddings before the backbone). That
    hard-coded float32 promotes the hidden states and breaks layer-norm under a
    bf16 ``dtype_override``. This module instead returns zeros in the model's
    current dtype (tracked via a buffer that ``.to(dtype)`` casts along with the
    rest of the model).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("_dtype_probe", torch.zeros(1), persistent=False)

    def forward(self, position_ids):
        return torch.zeros(
            position_ids.shape[0],
            position_ids.shape[1],
            self.dim,
            dtype=self._dtype_probe.dtype,
            device=position_ids.device,
        )


class XttsGptCore(nn.Module):
    """Single static forward pass over the XTTS-v2 GPT-2 transformer core.

    Holds the autoregressive GPT backbone (the compute-dominant 379M-param
    ``GPT2Model``) together with the final LayerNorm and mel head. Inputs are the
    precomputed concatenated embeddings ``[cond_latents, text_emb, mel_emb]`` of
    shape ``(batch, seq, model_dim)``; the output is mel-token logits of shape
    ``(batch, seq, num_audio_tokens)``.
    """

    def __init__(self, gpt: nn.Module):
        super().__init__()
        # gpt.gpt is the HF GPT2Model backbone (token/positional embeddings are
        # bypassed inside XTTS; the model is driven purely by inputs_embeds).
        self.gpt = gpt.gpt
        self.final_norm = gpt.final_norm
        self.mel_head = gpt.mel_head
        # Make the (zeroed) positional embedding dtype-aware so a bf16 override
        # doesn't get promoted back to float32 inside the backbone.
        self.gpt.wpe = _NullPositionEmbeddings(self.gpt.config.n_embd)

    def forward(self, inputs_embeds):
        hidden = self.gpt(inputs_embeds=inputs_embeds, return_dict=True).last_hidden_state
        hidden = self.final_norm(hidden)
        return self.mel_head(hidden)


class ModelVariant(StrEnum):
    """Available XTTS-v2 model variants."""

    V2 = "v2"


class ModelLoader(ForgeModel):
    """XTTS-v2 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.V2: ModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2

    # Fixed input sizing for a static forward pass.
    SAMPLE_TEXT = (
        "It took me quite a long time to develop a voice, and now that I "
        "have it I am not going to be silent."
    )
    SPEAKER = "Claribel Dervla"  # a built-in speaker shipped in speakers_xtts.pth
    TEXT_LEN = 32
    MEL_LEN = 32

    _DOWNLOAD_PATTERNS = [
        "config.json",
        "model.pth",
        "vocab.json",
        "speakers_xtts.pth",
        "mel_stats.pth",
        "dvae.pth",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._xtts = None
        self._ckpt_dir = None

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
        """Download the checkpoint and build the full coqui Xtts model (cached)."""
        if self._xtts is not None:
            return self._xtts

        _install_transformers_compat_shim()
        from huggingface_hub import snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        name = self._variant_config.pretrained_model_name
        ckpt_dir = snapshot_download(name, allow_patterns=self._DOWNLOAD_PATTERNS)

        config = XttsConfig()
        config.load_json(os.path.join(ckpt_dir, "config.json"))
        xtts = Xtts.init_from_config(config)
        xtts.load_checkpoint(
            config, checkpoint_dir=ckpt_dir, eval=True, use_deepspeed=False
        )
        xtts.eval()

        self._xtts = xtts
        self._ckpt_dir = ckpt_dir
        return xtts

    def load_model(self, dtype_override=None):
        """Load the XTTS-v2 GPT transformer core.

        Args:
            dtype_override: Optional torch.dtype to cast the model to.

        Returns:
            torch.nn.Module: XttsGptCore wrapping the GPT2 backbone + mel head.
        """
        xtts = self._load_xtts()
        model = XttsGptCore(xtts.gpt).eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        """Build the precomputed ``inputs_embeds`` for the GPT transformer core.

        The conditioning latent comes from a built-in speaker; the text and mel
        embeddings are computed on host from a sample sentence and a fixed set of
        mel start tokens. The three are concatenated into a single static-length
        sequence and fed to the GPT backbone.

        Args:
            dtype_override: Optional torch.dtype to cast the inputs to.

        Returns:
            dict: {"inputs_embeds": Tensor of shape (1, cond+text+mel, model_dim)}
        """
        xtts = self._load_xtts()
        gpt = xtts.gpt

        # Speaker conditioning latent (precomputed, shipped with the model).
        speakers = torch.load(
            os.path.join(self._ckpt_dir, "speakers_xtts.pth"),
            map_location="cpu",
            weights_only=False,
        )
        cond_latents = speakers[self.SPEAKER]["gpt_cond_latent"].float()  # (1, 32, dim)

        # Text embeddings (token + learned positional), padded to a fixed length.
        ids = torch.IntTensor(
            xtts.tokenizer.encode(self.SAMPLE_TEXT, lang="en")
        ).unsqueeze(0)
        ids = torch.nn.functional.pad(
            ids[:, : self.TEXT_LEN],
            (0, max(0, self.TEXT_LEN - ids.shape[1])),
            value=gpt.stop_text_token,
        )

        with torch.no_grad():
            text_emb = gpt.text_embedding(ids) + gpt.text_pos_embedding(ids)

            # Mel embeddings for a fixed set of start tokens.
            codes = torch.full(
                (1, self.MEL_LEN), gpt.start_audio_token, dtype=torch.long
            )
            mel_emb = gpt.mel_embedding(codes) + gpt.mel_pos_embedding(codes)

        inputs_embeds = torch.cat(
            [cond_latents, text_emb.float(), mel_emb.float()], dim=1
        )

        if dtype_override is not None:
            inputs_embeds = inputs_embeds.to(dtype_override)

        return {"inputs_embeds": inputs_embeds}
