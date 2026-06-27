# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is a StyleTTS2-based TTS model: a PL-BERT (ALBERT) text encoder feeds a
prosody/duration predictor (bidirectional LSTMs), whose predicted durations
expand the encoded phonemes into an alignment, which an ISTFTNet vocoder turns
into a waveform. The model is loaded through the upstream ``kokoro`` package;
``KModelForONNX`` exposes the clean tensor-in / tensor-out forward used here:

    forward(input_ids, ref_s, speed) -> (waveform, pred_dur)
"""
from typing import Optional

import torch

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


class ModelVariant(StrEnum):
    """Available Kokoro model variants."""

    BASE_82M = "82m"


class ModelLoader(ForgeModel):
    """Kokoro-82M model loader implementation for text-to-speech tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE_82M: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_82M

    # A fixed IPA phoneme string (keys of the model's vocab) so inputs are
    # reproducible without a G2P / espeak-ng dependency at load time.
    DEFAULT_PHONEMES = (
        "həlˈoʊ wˈɜːld, ðˈɪs ɪz ɐ tˈɛst ʌv ðə kˈoʊkəɹoʊ tˌiːtˌiːˈɛs mˈɑːdəl."
    )
    # One of the bundled voice packs (voices/<name>.pt in the HF repo).
    DEFAULT_VOICE = "af_heart"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the Kokoro loader.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
        """
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="kokoro_82m",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Kokoro model wrapped for tensor I/O.

        Args:
            dtype_override: Optional torch.dtype to cast the model weights to.

        Returns:
            torch.nn.Module: KModelForONNX whose forward is
                (input_ids, ref_s, speed) -> (waveform, pred_dur).
        """
        from kokoro import KModel
        from kokoro.model import KModelForONNX

        repo_id = self._variant_config.pretrained_model_name
        kmodel = KModel(repo_id=repo_id).eval()
        model = KModelForONNX(kmodel).eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._model = model
        return model

    def _get_vocab(self):
        """Return the phoneme->id vocab, reusing a loaded model if present."""
        if self._model is not None:
            return self._model.kmodel.vocab
        import json

        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name, filename="config.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)["vocab"]

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Kokoro model.

        Builds phoneme ``input_ids`` from a fixed IPA string and selects the
        matching style reference vector from a bundled voice pack.

        Args:
            dtype_override: Optional torch.dtype to cast the float inputs to.

        Returns:
            dict: {"input_ids", "ref_s", "speed"} for KModelForONNX.forward.
        """
        from huggingface_hub import hf_hub_download

        vocab = self._get_vocab()
        ids = [vocab.get(p) for p in self.DEFAULT_PHONEMES]
        ids = [i for i in ids if i is not None]
        # Wrap with BOS/EOS (token id 0), matching KModel.forward.
        input_ids = torch.LongTensor([[0, *ids, 0]])

        voice_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=f"voices/{self.DEFAULT_VOICE}.pt",
        )
        voice_pack = torch.load(voice_path, weights_only=True)
        # Voice packs are indexed by phoneme count (see KPipeline.infer).
        ref_s = voice_pack[len(ids) - 1]  # shape [1, 256]

        if dtype_override is not None:
            ref_s = ref_s.to(dtype_override)

        return {
            "input_ids": input_ids,
            "ref_s": ref_s,
            "speed": 1.0,
        }
