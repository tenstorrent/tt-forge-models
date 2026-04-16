# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ming-omni-tts model loader implementation for text-to-speech tasks.
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


class MingOmniTTSLLMWrapper(nn.Module):
    """Wrapper around the Ming-omni-tts LLM backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces logits for speech token prediction.
    """

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, inputs_embeds):
        outputs = self.llm(inputs_embeds=inputs_embeds, use_cache=False)
        return outputs.logits


class ModelVariant(StrEnum):
    """Available Ming-omni-tts model variants."""

    MING_OMNI_TTS_0_5B = "0.5B"


class ModelLoader(ForgeModel):
    """Ming-omni-tts model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MING_OMNI_TTS_0_5B: ModelConfig(
            pretrained_model_name="inclusionAI/Ming-omni-tts-0.5B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MING_OMNI_TTS_0_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Ming-omni-tts",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import hf_hub_download
        import json
        from transformers import Qwen2ForCausalLM, Qwen2Config

        # The HF repo is missing custom code files (configuration_bailingmm.py),
        # so we cannot use AutoModel. Instead, load the LLM backbone directly
        # using the Qwen2 config embedded in the model's config.json.
        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "config.json"
        )
        with open(config_path) as f:
            full_config = json.load(f)

        llm_config = Qwen2Config(**full_config["llm_config"])
        llm = Qwen2ForCausalLM(llm_config)
        llm.to(dtype=dtype_override or torch.float32)

        model = MingOmniTTSLLMWrapper(llm)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # LLM backbone hidden_size=896, use a short sequence of embeddings
        inputs_embeds = torch.randn(1, 32, 896, dtype=dtype)
        return (inputs_embeds,)
