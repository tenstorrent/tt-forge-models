# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Coqui XTTS-v2 model loader implementation for text-to-speech tasks.

XTTS-v2 is a multi-component voice-cloning TTS pipeline (Coqui `TTS` library):
  * an autoregressive GPT-2 transformer (30 layers, d_model=1024, 16 heads) that
    predicts discrete audio tokens from text + speaker-conditioning embeddings,
  * a HiFi-GAN decoder that converts GPT latents to a 24 kHz waveform,
  * a discrete VAE (DVAE) used for the mel-token codebook.

The compute-dominant component is the GPT-2 autoregressive backbone (~379M of the
~441M GPT params). This loader brings up that inner transformer as a single forward
pass over `inputs_embeds`, mirroring how SeamlessM4T's loader returns just its text
decoder submodule. The HiFi-GAN vocoder is a neural-vocoder architecture (conv +
weight-norm) that the static-shape device path does not yet support, so it is not
the bring-up target here.
"""
import functools
import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _apply_transformers_compat_shims():
    """Vendored compatibility shims for running coqui-tts against transformers 5.x.

    coqui-tts 0.27.5 imports ``isin_mps_friendly`` from ``transformers.pytorch_utils``,
    which was removed in transformers 5.x. We restore it (a thin wrapper over
    ``torch.isin``) so the XTTS import chain succeeds without pinning transformers
    back to a 4.x release that would conflict with the rest of the tt-xla stack.
    """
    import transformers.pytorch_utils as ptu

    if not hasattr(ptu, "isin_mps_friendly"):

        def isin_mps_friendly(elements, test_elements):
            return torch.isin(elements, test_elements)

        ptu.isin_mps_friendly = isin_mps_friendly


def _typed_null_position_embeddings(position_ids, dim, dtype):
    """Dtype-correct replacement for XTTS's ``null_position_embeddings``.

    XTTS zeroes out GPT-2's built-in positional embedding (it supplies its own
    learned positional embedding upstream) by swapping ``gpt.wpe`` for a stub that
    returns ``torch.zeros(...)`` in the default float32 dtype. When the model runs
    in bf16, those fp32 zeros are added to the bf16 hidden states and upcast them to
    fp32, which then mismatches the bf16 LayerNorm weights ("mixed dtype" error).
    This version returns zeros in the model's dtype so the whole stack stays in one
    dtype; the values are identically zero, so numerics are unchanged.
    """
    return torch.zeros((*position_ids.shape, dim), device=position_ids.device, dtype=dtype)


class ModelVariant(StrEnum):
    """Available XTTS model variants."""

    XTTS_V2 = "v2"


class ModelLoader(ForgeModel):
    """Coqui XTTS-v2 loader — brings up the GPT-2 autoregressive backbone."""

    _VARIANTS = {
        ModelVariant.XTTS_V2: ModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XTTS_V2

    # Fixed sample utterance for reproducible inputs.
    DEFAULT_TEXT = "Hello world, this is a test of the text to speech model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.xtts = None
        self.gpt_module = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="xtts",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_full_xtts(self):
        """Download checkpoint/config/vocab and build the full Xtts model."""
        from huggingface_hub import hf_hub_download

        _apply_transformers_compat_shims()
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        repo = self._variant_config.pretrained_model_name
        ckpt = hf_hub_download(repo, "model.pth")
        cfg_path = hf_hub_download(repo, "config.json")
        vocab = hf_hub_download(repo, "vocab.json")

        config = XttsConfig()
        config.load_json(cfg_path)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=ckpt,
            vocab_path=vocab,
            use_deepspeed=False,
            eval=True,
        )
        model.eval()

        self.xtts = model
        self.gpt_module = model.gpt
        self.tokenizer = model.tokenizer
        return model

    def load_model(self, dtype_override=None):
        """Load XTTS-v2 and return its inner GPT-2 autoregressive transformer.

        Args:
            dtype_override: Optional torch.dtype applied to the returned module.

        Returns:
            torch.nn.Module: The GPT2Model transformer backbone (model.gpt.gpt).
        """
        if self.xtts is None:
            self._load_full_xtts()

        gpt2 = self.gpt_module.gpt  # transformers.GPT2Model

        if dtype_override is not None:
            gpt2 = gpt2.to(dtype_override)

        # Keep GPT-2's (zeroed) positional embedding in the model's dtype so the
        # hidden states are not silently upcast to fp32 (see helper docstring).
        model_dtype = next(gpt2.parameters()).dtype
        gpt2.wpe = functools.partial(
            _typed_null_position_embeddings, dim=gpt2.config.n_embd, dtype=model_dtype
        )

        return gpt2

    def load_inputs(self, dtype_override=None):
        """Build faithful `inputs_embeds` from a sample utterance.

        The text is tokenized with XTTS's VoiceBpeTokenizer and embedded with the
        GPT module's text embedding + learned text positional embedding — exactly
        the text portion of the sequence XTTS feeds its GPT-2 backbone.

        Args:
            dtype_override: Optional torch.dtype to cast the inputs to.

        Returns:
            dict: {"inputs_embeds": tensor of shape [1, seq, 1024]}
        """
        if self.gpt_module is None:
            self._load_full_xtts()

        token_ids = self.tokenizer.encode(self.DEFAULT_TEXT, lang="en")
        ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            text_emb = self.gpt_module.text_embedding(ids)
            pos_emb = self.gpt_module.text_pos_embedding(text_emb)
            inputs_embeds = text_emb + pos_emb

        if dtype_override is not None:
            inputs_embeds = inputs_embeds.to(dtype_override)

        return {"inputs_embeds": inputs_embeds}
