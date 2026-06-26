# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Coqui XTTS-v2 model loader implementation for text-to-speech tasks.

XTTS-v2 is a multi-component TTS pipeline. This loader brings up its
compute-dominant GPT-2 autoregressive backbone (the LM that predicts text +
audio-code tokens) as a single forward pass. See ``src/xtts_backbone.py`` for
the architecture notes and the conditioning contract.
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
from .src.xtts_backbone import (
    XttsGptBackbone,
    load_xtts_model,
    MODEL_DIM,
    NUM_AUDIO_TOKENS,
    NUM_COND_LATENTS,
    NUM_TEXT_TOKENS,
)


class ModelVariant(StrEnum):
    """Available XTTS variants."""

    V2 = "v2"


class ModelLoader(ForgeModel):
    """Coqui XTTS-v2 loader: brings up the GPT-2 LM backbone of the TTS pipeline."""

    _VARIANTS = {
        ModelVariant.V2: ModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2

    # Fixed input lengths for the single forward pass (kept small/static for the
    # static-shape device path). text/mel are token-id sequences; cond is the
    # 32 perceiver-resampler conditioning latents.
    TEXT_LEN = 16
    MEL_LEN = 32
    COND_LEN = NUM_COND_LATENTS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._xtts = None

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

    def load_model(self, dtype_override=None, **kwargs):
        """Load XTTS-v2 and return the GPT-2 backbone wrapped as an nn.Module.

        Args:
            dtype_override: Optional torch.dtype applied to the backbone weights
                (the runner passes torch.bfloat16).
        """
        xtts, _ = load_xtts_model()
        self._xtts = xtts

        model = XttsGptBackbone(xtts.gpt)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        """Build a fixed-shape batch of sample inputs for the GPT backbone.

        text_inputs / audio_codes are integer token ids; cond_latents are the
        conditioning embeddings (float) that the speaker/perceiver encoder would
        produce from reference audio.
        """
        # Deterministic so the CPU golden and the PoC artifact reproduce.
        g = torch.Generator().manual_seed(0)

        text_inputs = torch.randint(
            0, NUM_TEXT_TOKENS, (1, self.TEXT_LEN), generator=g, dtype=torch.long
        )
        audio_codes = torch.randint(
            0, NUM_AUDIO_TOKENS, (1, self.MEL_LEN), generator=g, dtype=torch.long
        )
        cond_latents = torch.randn(1, self.COND_LEN, MODEL_DIM, generator=g)
        if dtype_override is not None:
            cond_latents = cond_latents.to(dtype_override)

        return {
            "text_inputs": text_inputs,
            "audio_codes": audio_codes,
            "cond_latents": cond_latents,
        }
