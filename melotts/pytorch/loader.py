# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MeloTTS-French model loader implementation for text-to-speech tasks.
"""
import json
import os
import sys

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
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


class HParams:
    """Lightweight hyperparameter container that supports attribute access."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


class MeloTTSWrapper(nn.Module):
    """Wrapper around MeloTTS SynthesizerTrn to expose infer as forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, sid, tones, lang_ids, bert, ja_bert):
        audio = self.model.infer(
            x,
            x_lengths,
            sid,
            tones,
            lang_ids,
            bert,
            ja_bert,
        )[0]
        return audio


class ModelVariant(StrEnum):
    """Available MeloTTS model variants."""

    MELOTTS_FRENCH = "French"


class ModelLoader(ForgeModel):
    """MeloTTS-French model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MELOTTS_FRENCH: ModelConfig(
            pretrained_model_name="myshell-ai/MeloTTS-French",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MELOTTS_FRENCH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._hps = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MeloTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import melo

        melo_dir = os.path.dirname(melo.__file__)
        if melo_dir not in sys.path:
            sys.path.insert(0, melo_dir)

        from melo.models import SynthesizerTrn

        repo_id = self._variant_config.pretrained_model_name
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pth")

        with open(config_path) as f:
            data = json.load(f)
        hps = HParams(**data)
        self._hps = hps

        model = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=hps.num_tones,
            num_languages=hps.num_languages,
            **dict(hps.model.items()),
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return MeloTTSWrapper(model)

    def load_inputs(self, dtype_override=None):
        seq_len = 50
        num_symbols = len(self._hps.symbols) if self._hps else 219
        x = torch.randint(0, num_symbols, (1, seq_len), dtype=torch.long)
        x_lengths = torch.tensor([seq_len], dtype=torch.long)
        sid = torch.tensor([0], dtype=torch.long)
        tones = torch.zeros(1, seq_len, dtype=torch.long)
        lang_ids = torch.zeros(1, seq_len, dtype=torch.long)
        bert = torch.zeros(1, 1024, seq_len)
        ja_bert = torch.zeros(1, 768, seq_len)
        return x, x_lengths, sid, tones, lang_ids, bert, ja_bert
