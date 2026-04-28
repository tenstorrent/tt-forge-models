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
        self.model_name = "ACE-Step/ACE-Step-v1-3.5B"
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
        from acestep.models.ace_step_transformer import ACEStepTransformer2DModel

        self.model = ACEStepTransformer2DModel.from_pretrained(
            self.model_name,
            subfolder="ace_step_transformer",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        return self.model

    def load_inputs(self, batch_size=1):
        # ACEStepTransformer2DModel.forward signature:
        #   hidden_states: (batch, in_channels=8, height=16, width)
        #   attention_mask: (batch, width)  float mask for self-attention
        #   encoder_text_hidden_states: (batch, text_len, 768)
        #   text_attention_mask: (batch, text_len)  float mask
        #   speaker_embeds: (batch, 512)
        #   lyric_token_idx: (batch, lyric_len)  long
        #   lyric_mask: (batch, lyric_len)  float mask
        #   timestep: (batch,)
        #
        # width >= 32 required for SDPA k_chunk_size constraint.
        width = 256
        text_len = 64
        lyric_len = 64

        hidden_states = torch.randn(batch_size, 8, 16, width, dtype=torch.bfloat16)
        attention_mask = torch.ones(batch_size, width, dtype=torch.float32)
        encoder_text_hidden_states = torch.randn(
            batch_size, text_len, 768, dtype=torch.bfloat16
        )
        text_attention_mask = torch.ones(batch_size, text_len, dtype=torch.float32)
        speaker_embeds = torch.zeros(batch_size, 512, dtype=torch.bfloat16)
        # lyric_token_idx: vocab size is 6681
        lyric_token_idx = torch.zeros(batch_size, lyric_len, dtype=torch.long)
        lyric_mask = torch.ones(batch_size, lyric_len, dtype=torch.float32)
        timestep = torch.full((batch_size,), 500.0, dtype=torch.float32)

        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "encoder_text_hidden_states": encoder_text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "speaker_embeds": speaker_embeds,
            "lyric_token_idx": lyric_token_idx,
            "lyric_mask": lyric_mask,
            "timestep": timestep,
            "return_dict": False,
        }

    def unpack_forward_output(self, output):
        # return_dict=False returns (sample_tensor, proj_losses_list)
        return output[0]
