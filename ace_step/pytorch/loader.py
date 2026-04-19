# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 model loader implementation for music generation tasks.
"""

import torch

from ...base import ForgeModel
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 model loader implementation for music generation."""

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = "ACE-Step/Ace-Step1.5"
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="ACE-Step 1.5",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        import os

        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        repo_path = snapshot_download(self.model_name)
        model_path = os.path.join(repo_path, "acestep-v15-turbo")
        full_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs,
        )
        self.model = full_model.decoder
        self.model.eval()
        return self.model

    def load_inputs(self, batch_size=1):
        seq_len = 64
        enc_seq_len = 32
        hidden_dim = 64
        context_dim = 128
        encoder_dim = 2048
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        timestep = torch.rand(batch_size)
        timestep_r = torch.rand(batch_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        encoder_hidden_states = torch.randn(batch_size, enc_seq_len, encoder_dim)
        encoder_attention_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.long)
        context_latents = torch.randn(batch_size, seq_len, context_dim)
        return (
            hidden_states,
            timestep,
            timestep_r,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
        )
