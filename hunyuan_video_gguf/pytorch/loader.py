# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanVideo GGUF model loader implementation.

Loads GGUF-quantized HunyuanVideo T2V 720p transformers from
city96/HunyuanVideo-gguf via diffusers' HunyuanVideoTransformer3DModel
from_single_file. HunyuanVideo is a 13B parameter text-to-video
diffusion model from Tencent.

Repository:
- https://huggingface.co/city96/HunyuanVideo-gguf
"""
from typing import Optional

import torch
from diffusers import HunyuanVideoTransformer3DModel

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

GGUF_BASE_URL = "https://huggingface.co/city96/HunyuanVideo-gguf/blob/main"


class ModelVariant(StrEnum):
    """Available HunyuanVideo GGUF model variants."""

    Q4_K_S = "Q4_K_S"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """HunyuanVideo GGUF model loader for text-to-video generation."""

    _VARIANTS = {
        ModelVariant.Q4_K_S: ModelConfig(
            pretrained_model_name="city96/HunyuanVideo-gguf",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="city96/HunyuanVideo-gguf",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K_S: "hunyuan-video-t2v-720p-Q4_K_S.gguf",
        ModelVariant.Q8_0: "hunyuan-video-t2v-720p-Q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HunyuanVideo GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = self._GGUF_FILES[self._variant]
        gguf_url = f"{GGUF_BASE_URL}/{gguf_file}"

        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = HunyuanVideoTransformer3DModel.from_single_file(
            gguf_url,
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # HunyuanVideo transformer dimensions. Use latent sizes for a
        # sub-720p frame to keep synthetic inputs compact for compile tests.
        num_channels = config.in_channels
        num_frames = 9
        height = 60
        width = 104
        seq_len = 256

        # Latent video tensor: [batch, channels, frames, height, width]
        hidden_states = torch.randn(
            batch_size, num_channels, num_frames, height, width, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        # Text encoder hidden states: [batch, seq_len, text_embed_dim]
        encoder_hidden_states = torch.randn(
            batch_size, seq_len, config.text_embed_dim, dtype=dtype
        )

        # Encoder attention mask: [batch, seq_len]
        encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Pooled projections: [batch, pooled_projection_dim]
        pooled_projections = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "pooled_projections": pooled_projections,
        }

        return inputs
