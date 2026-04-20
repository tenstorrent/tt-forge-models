# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CosyVoice2 0.5B model loader implementation for text-to-speech tasks.
"""
import os

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


class CosyVoice2Qwen2Wrapper(nn.Module):
    """Wrapper around the CosyVoice2 Qwen2 LLM backbone.

    CosyVoice2 uses a Qwen2ForCausalLM (shipped in the CosyVoice-BlankEN
    subfolder of the repo) as the text/semantic-token language model.
    This wrapper exposes a clean forward pass that takes pre-computed
    input embeddings and produces logits.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds):
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        return outputs.logits


class ModelVariant(StrEnum):
    """Available CosyVoice2 model variants."""

    COSYVOICE2_0_5B = "0.5B"


class ModelLoader(ForgeModel):
    """CosyVoice2 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.COSYVOICE2_0_5B: ModelConfig(
            pretrained_model_name="FunAudioLLM/CosyVoice2-0.5B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COSYVOICE2_0_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CosyVoice2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import snapshot_download
        from transformers import Qwen2ForCausalLM

        local_dir = snapshot_download(
            repo_id=self._variant_config.pretrained_model_name,
            allow_patterns=["CosyVoice-BlankEN/*"],
        )
        backbone_dir = os.path.join(local_dir, "CosyVoice-BlankEN")

        model = Qwen2ForCausalLM.from_pretrained(
            backbone_dir,
            torch_dtype=dtype_override or torch.float32,
        )
        model.eval()
        return CosyVoice2Qwen2Wrapper(model)

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Qwen2 backbone in CosyVoice-BlankEN: hidden_size=896.
        inputs_embeds = torch.randn(1, 32, 896, dtype=dtype)
        return (inputs_embeds,)
