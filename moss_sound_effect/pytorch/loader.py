# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MOSS-SoundEffect model loader implementation for text-to-audio generation.
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


class MossSoundEffectWrapper(nn.Module):
    """Wrapper around the MOSS-SoundEffect backbone.

    Exposes a clean forward pass through the MossTTSDelay language backbone,
    producing text-head logits from multi-channel audio+text input.
    """

    def __init__(self, model):
        super().__init__()
        self.language_model = model.language_model
        self.emb_ext = model.emb_ext
        self.lm_text_head = model.lm_heads[0]
        self.n_vq = model.config.n_vq

    def forward(self, input_ids, attention_mask):
        # input_ids: (B, S, 1 + n_vq) — channel 0 is text, channels 1..n_vq are audio VQ
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids[..., 0])
        for i, embed_layer in enumerate(self.emb_ext):
            inputs_embeds = inputs_embeds + embed_layer(input_ids[..., i + 1])
        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        return self.lm_text_head(outputs.last_hidden_state)


class ModelVariant(StrEnum):
    """Available MOSS-SoundEffect model variants."""

    MOSS_SOUND_EFFECT = "moss-sound-effect"


class ModelLoader(ForgeModel):
    """MOSS-SoundEffect model loader for text-to-audio generation."""

    _VARIANTS = {
        ModelVariant.MOSS_SOUND_EFFECT: ModelConfig(
            pretrained_model_name="OpenMOSS-Team/MOSS-SoundEffect",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOSS_SOUND_EFFECT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MOSS-SoundEffect",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        full_model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=dtype_override or torch.float32,
            **kwargs,
        )
        model = MossSoundEffectWrapper(full_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        n_vq = 16
        # Channel 0: text tokens; channels 1..n_vq: audio VQ tokens
        input_ids = torch.cat(
            [
                torch.randint(0, 1000, (1, 32, 1)),
                torch.randint(0, 1024, (1, 32, n_vq)),
            ],
            dim=-1,
        )
        attention_mask = torch.ones(1, 32, dtype=torch.long)
        return (input_ids, attention_mask)
