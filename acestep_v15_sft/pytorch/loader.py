# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step v1.5 SFT model loader implementation for text-to-music generation.

Loads ACE-Step/acestep-v15-sft, the supervised fine-tuned DiT (Diffusion
Transformer) checkpoint of the ACE-Step v1.5 music foundation model. The full
AceStepConditionGenerationModel forward is training-oriented, so we expose
only the DiT decoder submodule for inference.
"""

from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ACE-Step v1.5 SFT model variants."""

    SFT = "sft"


class ModelLoader(ForgeModel):
    """ACE-Step v1.5 SFT model loader implementation for text-to-music generation."""

    _VARIANTS = {
        ModelVariant.SFT: ModelConfig(
            pretrained_model_name="ACE-Step/acestep-v15-sft",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SFT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ACE-Step v1.5 SFT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": False}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        full_model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        self.model = full_model.decoder
        self.model.eval()
        return self.model

    def load_inputs(self, *, dtype_override=None, batch_size=1):
        # DiT decoder inputs. Channel dims follow the ACE-Step v1.5 config:
        # in_channels = audio_acoustic_hidden_dim (64) + context_dim (128) = 192
        # encoder hidden size = 2048
        seq_len = 64
        enc_seq_len = 32
        hidden_dim = 64
        context_dim = 128
        encoder_dim = 2048

        dtype = dtype_override or torch.float32
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype)
        timestep = torch.rand(batch_size, dtype=dtype)
        timestep_r = torch.rand(batch_size, dtype=dtype)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        encoder_hidden_states = torch.randn(
            batch_size, enc_seq_len, encoder_dim, dtype=dtype
        )
        encoder_attention_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.long)
        context_latents = torch.randn(batch_size, seq_len, context_dim, dtype=dtype)
        return (
            hidden_states,
            timestep,
            timestep_r,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
        )
