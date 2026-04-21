# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dia2 1B streaming dialogue text-to-speech model loader implementation.
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


class Dia2StepWrapper(nn.Module):
    """Wrap Dia2Model.step_text to expose a single decode step as a clean forward.

    The Dia2 transformer decodes one multi-stream token per call (text + audio
    codebooks) and returns the hidden state, action logits, and codebook-0
    logits. A persistent KV-cache ``state`` is allocated at construction time.
    """

    def __init__(self, model, max_steps: int):
        super().__init__()
        self.model = model
        self.state = model.init_state(
            batch_size=1,
            device=torch.device("cpu"),
            max_steps=max_steps,
        )

    def forward(self, tokens, positions):
        hidden, action, cb0 = self.model.step_text(tokens, positions, self.state)
        return hidden, action, cb0


class ModelVariant(StrEnum):
    """Available Dia2 model variants."""

    DIA2_1B = "Dia2-1B"


class ModelLoader(ForgeModel):
    """Dia2 1B streaming dialogue text-to-speech model loader implementation."""

    _VARIANTS = {
        ModelVariant.DIA2_1B: ModelConfig(
            pretrained_model_name="nari-labs/Dia2-1B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIA2_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._channels = None
        self._max_steps = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Dia2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Dia2 model backbone on CPU and wrap a single text decode step."""
        from dia2.assets import resolve_assets
        from dia2.config import load_config
        from dia2.core.model import Dia2Model
        from dia2.core.precision import Precision
        from dia2.runtime.context import load_file_into_model

        pretrained_model_name = self._variant_config.pretrained_model_name

        bundle = resolve_assets(
            repo=pretrained_model_name,
            config_path=None,
            weights_path=None,
        )
        config = load_config(bundle.config_path)

        compute_dtype = dtype_override if dtype_override is not None else torch.float32
        precision = Precision(compute=compute_dtype, logits=torch.float32)

        device = torch.device("cpu")
        model = Dia2Model(config, precision, device=device)
        load_file_into_model(model, bundle.weights_path, device="cpu")
        model.eval()

        self._channels = config.data.channels
        self._max_steps = config.runtime.max_context_steps

        return Dia2StepWrapper(model, max_steps=self._max_steps)

    def load_inputs(self, dtype_override=None):
        """Provide a single-step multi-stream token batch and position index.

        ``tokens`` has shape [B=1, channels, T=1] where channel 0/1 are text
        streams and channels 2..(channels-1) are audio codebooks. ``positions``
        is the decode step index used for RoPE / KV-cache writes.
        """
        tokens = torch.zeros(1, self._channels, 1, dtype=torch.long)
        positions = torch.zeros(1, 1, dtype=torch.long)
        return (tokens, positions)
