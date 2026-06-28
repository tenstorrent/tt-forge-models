# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is a StyleTTS2 + ISTFTNet text-to-speech model. It is loaded through the
upstream ``kokoro`` PyPI package (not ``transformers``); ``KModelForONNX`` wraps
``KModel`` to give a clean tensor-in/tensor-out forward:
``(input_ids, ref_s, speed) -> (waveform, pred_dur)``.
"""
from typing import Optional

import torch

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


class ModelVariant(StrEnum):
    """Available Kokoro-82M model variants."""

    KOKORO_82M = "82m"


class ModelLoader(ForgeModel):
    """Kokoro-82M model loader implementation for text-to-speech tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.KOKORO_82M: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.KOKORO_82M

    # Default voice pack used to build the reference style vector.
    DEFAULT_VOICE = "af_heart"

    # Fixed IPA phoneme string ("Hello world") so no espeak/G2P dependency is
    # needed at inference time; mapped to ids through the model's own vocab.
    DEFAULT_PHONEMES = "həlˈoʊ wˈɜːld"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None
        self._input_ids = None

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
        """Load the Kokoro-82M model wrapped for a clean tensor forward.

        Returns ``KModelForONNX`` whose ``forward(input_ids, ref_s, speed)``
        returns ``(waveform, pred_dur)``.
        """
        from kokoro import KModel
        from kokoro.model import KModelForONNX

        model_name = self._variant_config.pretrained_model_name
        kmodel = KModel(model_name).eval()
        model = KModelForONNX(kmodel).eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Build sample inputs: phoneme ``input_ids`` and a style ``ref_s``.

        ``input_ids`` is the fixed IPA phoneme string mapped through the model's
        vocab and wrapped with the boundary token ``0``. ``ref_s`` is taken from
        the ``af_heart`` voice pack, indexed by the phoneme count (the style
        vector Kokoro selects per utterance length).
        """
        from huggingface_hub import hf_hub_download

        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        vocab = self._model.kmodel.vocab
        ids = [vocab.get(c) for c in self.DEFAULT_PHONEMES]
        ids = [i for i in ids if i is not None]
        input_ids = torch.LongTensor([[0, *ids, 0]])

        model_name = self._variant_config.pretrained_model_name
        voice_path = hf_hub_download(model_name, f"voices/{self.DEFAULT_VOICE}.pt")
        voice_pack = torch.load(voice_path, weights_only=True)
        ref_s = voice_pack[len(ids)]

        if dtype_override is not None:
            ref_s = ref_s.to(dtype_override)

        self._input_ids = input_ids
        return {"input_ids": input_ids, "ref_s": ref_s, "speed": 1}
