# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MeloTTS-French model loader implementation for text-to-speech tasks.
"""
import os
import subprocess
import sys

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
        self._tts = None

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

    @staticmethod
    def _ensure_unidic():
        import unidic

        dicdir = unidic.DICDIR
        if not os.path.isfile(os.path.join(dicdir, "mecabrc")):
            subprocess.check_call([sys.executable, "-m", "unidic", "download"])

    @staticmethod
    def _patch_melo_transforms():
        import math
        import melo.transforms as transforms

        _orig = transforms.unconstrained_rational_quadratic_spline

        def _patched(
            inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=False,
            tails="linear",
            tail_bound=1.0,
            min_bin_width=transforms.DEFAULT_MIN_BIN_WIDTH,
            min_bin_height=transforms.DEFAULT_MIN_BIN_HEIGHT,
            min_derivative=transforms.DEFAULT_MIN_DERIVATIVE,
        ):
            from torch.nn import functional as F

            inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
            outside_interval_mask = ~inside_interval_mask

            outputs = torch.zeros_like(inputs)
            logabsdet = torch.zeros_like(inputs)

            if tails == "linear":
                unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
                constant = math.log(math.exp(1 - float(min_derivative)) - 1)
                unnormalized_derivatives[..., 0] = constant
                unnormalized_derivatives[..., -1] = constant

                outputs[outside_interval_mask] = inputs[outside_interval_mask]
                logabsdet[outside_interval_mask] = 0
            else:
                raise RuntimeError(f"{tails} tails are not implemented.")

            (
                outputs[inside_interval_mask],
                logabsdet[inside_interval_mask],
            ) = transforms.rational_quadratic_spline(
                inputs=inputs[inside_interval_mask],
                unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
                unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
                unnormalized_derivatives=unnormalized_derivatives[
                    inside_interval_mask, :
                ],
                inverse=inverse,
                left=-tail_bound,
                right=tail_bound,
                bottom=-tail_bound,
                top=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )

            return outputs, logabsdet

        transforms.unconstrained_rational_quadratic_spline = _patched

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_unidic()
        from melo.api import TTS

        self._tts = TTS(language="FR", device="cpu")
        self._patch_melo_transforms()
        model = MeloTTSWrapper(self._tts.model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        from melo import utils

        text = "Bonjour, ceci est un test."
        device = "cpu"
        language = self._tts.language

        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            text, language, self._tts.hps, device, self._tts.symbol_to_id
        )

        x = phones.unsqueeze(0)
        x_lengths = torch.LongTensor([phones.size(0)])
        sid = torch.LongTensor([list(self._tts.hps.data.spk2id.values())[0]])
        tones = tones.unsqueeze(0)
        lang_ids = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)

        return x, x_lengths, sid, tones, lang_ids, bert, ja_bert
