# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MOSS-TTS-Realtime model loader implementation for text-to-speech tasks.
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


class MossTTSRealtimeLanguageWrapper(nn.Module):
    """Wrapper around the MOSS-TTS-Realtime Qwen3 language backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces hidden states from the language model.
    """

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def forward(self, inputs_embeds):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        return outputs.last_hidden_state


class ModelVariant(StrEnum):
    """Available MOSS-TTS-Realtime model variants."""

    MOSS_TTS_REALTIME_1_7B = "1.7B"


class ModelLoader(ForgeModel):
    """MOSS-TTS-Realtime model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MOSS_TTS_REALTIME_1_7B: ModelConfig(
            pretrained_model_name="OpenMOSS-Team/MOSS-TTS-Realtime",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOSS_TTS_REALTIME_1_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MOSS-TTS-Realtime",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        repo = self._variant_config.pretrained_model_name
        config_class = get_class_from_dynamic_module(
            "configuration_mossttsrealtime.MossTTSRealtimeConfig",
            repo,
        )
        config = config_class.from_pretrained(repo)

        model_class = get_class_from_dynamic_module(
            "modeling_mossttsrealtime.MossTTSRealtime",
            repo,
        )
        full_model = model_class.from_pretrained(
            repo,
            config=config,
            torch_dtype=dtype_override or torch.float32,
        )
        model = MossTTSRealtimeLanguageWrapper(full_model.language_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Qwen3 language backbone hidden_size=2048, use a short sequence
        inputs_embeds = torch.randn(1, 32, 2048, dtype=dtype)
        return (inputs_embeds,)
