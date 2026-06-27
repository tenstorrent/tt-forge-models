# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is a StyleTTS2 + ISTFTNet text-to-speech model. It is loaded through the
upstream ``kokoro`` PyPI package (not transformers): the public weights live at
``hexgrad/Kokoro-82M`` on HuggingFace.

We wrap ``KModel`` in ``KModelForONNX``, which exposes a clean tensor-only
forward:

    (input_ids, ref_s, speed) -> (waveform, pred_dur)

``load_inputs`` builds the phoneme ``input_ids`` from a fixed IPA string mapped
through ``model.vocab`` (so no espeak / G2P runtime dependency is needed) and
selects the style/reference vector ``ref_s`` from the ``af_heart`` voice pack,
indexed by the phoneme count exactly as ``KPipeline`` does.
"""
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


class ModelVariant(StrEnum):
    """Available Kokoro-82M model variants."""

    BASE = "82m"


class ModelLoader(ForgeModel):
    """Kokoro-82M model loader implementation for text-to-speech tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Fixed IPA phoneme string ("Hello world, this is a test.") so the loader has
    # no runtime G2P / espeak dependency. Every glyph is present in KModel.vocab.
    DEFAULT_PHONEMES = "həlˈoʊ wˈɜːld, ðɪs ɪz ɐ tˈɛst."

    # Default voice pack on HuggingFace (hexgrad/Kokoro-82M/voices/af_heart.pt).
    DEFAULT_VOICE = "af_heart"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None
        self._vocab = None

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
        """Load Kokoro-82M wrapped for a tensor-only (ONNX-style) forward."""
        from kokoro import KModel
        from kokoro.model import KModelForONNX

        pretrained = self._variant_config.pretrained_model_name
        kmodel = KModel(repo_id=pretrained).eval()
        self._vocab = kmodel.vocab
        model = KModelForONNX(kmodel).eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Build (input_ids, ref_s, speed) inputs for Kokoro-82M.

        ``input_ids`` are the fixed IPA phonemes mapped through the model vocab
        and wrapped with the boundary token 0 at both ends, matching
        ``KModel.forward``. ``ref_s`` is selected from the voice pack by phoneme
        count, matching ``KPipeline.infer`` (``pack[len(phonemes) - 1]``).
        """
        from huggingface_hub import hf_hub_download

        if self._vocab is None:
            self.load_model(dtype_override=dtype_override)

        phonemes = self.DEFAULT_PHONEMES
        mapped = [self._vocab.get(p) for p in phonemes]
        mapped = [i for i in mapped if i is not None]
        input_ids = torch.LongTensor([[0, *mapped, 0]])

        pretrained = self._variant_config.pretrained_model_name
        voice_path = hf_hub_download(
            repo_id=pretrained, filename=f"voices/{self.DEFAULT_VOICE}.pt"
        )
        pack = torch.load(voice_path, weights_only=True)
        ref_s = pack[len(mapped) - 1]
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)

        if dtype_override is not None:
            ref_s = ref_s.to(dtype_override)

        return {"input_ids": input_ids, "ref_s": ref_s, "speed": 1.0}
