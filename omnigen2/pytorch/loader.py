# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OmniGen2 (OmniGen2/OmniGen2) model loader implementation.

OmniGen2 is a unified multimodal generation model capable of text-to-image
generation, image editing, and visual understanding tasks.

Available variants:
- OMNIGEN2: OmniGen2/OmniGen2 text-to-image generation
"""

import os
import sys
from typing import Optional

import torch
from huggingface_hub import snapshot_download

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


REPO_ID = "OmniGen2/OmniGen2"


class ModelVariant(StrEnum):
    """Available OmniGen2 model variants."""

    OMNIGEN2 = "OmniGen2"


class ModelLoader(ForgeModel):
    """OmniGen2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.OMNIGEN2: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.OMNIGEN2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self._cache_dir = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OmniGen2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer_class(self):
        self._cache_dir = snapshot_download(
            self._variant_config.pretrained_model_name,
            allow_patterns=["transformer/*"],
        )
        transformer_dir = os.path.join(self._cache_dir, "transformer")
        if transformer_dir not in sys.path:
            sys.path.insert(0, transformer_dir)
        from transformer_omnigen2 import OmniGen2Transformer2DModel

        return OmniGen2Transformer2DModel

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OmniGen2 transformer model.

        Returns:
            torch.nn.Module: The OmniGen2 transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        cls = self._load_transformer_class()
        self.transformer = cls.from_pretrained(
            self._cache_dir,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the OmniGen2 transformer.

        Returns:
            dict: Input tensors matching the transformer's forward signature.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        config = self.transformer.config

        height, width = 64, 64
        text_seq_len = 32

        hidden_states = torch.randn(
            batch_size, config.in_channels, height, width, dtype=dtype
        )
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        text_hidden_states = torch.randn(
            batch_size, text_seq_len, config.text_feat_dim, dtype=dtype
        )

        freqs_cis = self.transformer.rope_embedder.get_freqs_cis(
            axes_dim=tuple(config.axes_dim_rope),
            axes_lens=tuple(config.axes_lens),
            theta=10000,
        )

        text_attention_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "text_hidden_states": text_hidden_states,
            "freqs_cis": freqs_cis,
            "text_attention_mask": text_attention_mask,
            "return_dict": False,
        }
