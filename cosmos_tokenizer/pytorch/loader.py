# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Cosmos Tokenizer model loader implementation for continuous video tokenization.

Uses the cosmos-tokenizer package to construct the model architecture directly,
avoiding the gated HuggingFace repo download.
"""

import torch
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


class ModelVariant(StrEnum):
    """Available Cosmos Tokenizer model variants."""

    CV8X8X8_720P = "CV8x8x8-720p"


class ModelLoader(ForgeModel):
    """NVIDIA Cosmos Tokenizer model loader for continuous video tokenization."""

    _VARIANTS = {
        ModelVariant.CV8X8X8_720P: ModelConfig(
            pretrained_model_name="nvidia/Cosmos-Tokenize1-CV8x8x8-720p",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CV8X8X8_720P

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Cosmos-Tokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _import_cosmos_package():
        """Import from the pip-installed cosmos_tokenizer despite the local name clash."""
        import importlib
        import site
        import sys

        saved_modules = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "cosmos_tokenizer" or k.startswith("cosmos_tokenizer.")
        }
        saved_path = sys.path[:]
        try:
            sys.path = [
                p
                for p in sys.path
                if "site-packages" in p or p in site.getsitepackages()
            ]
            configs = importlib.import_module("cosmos_tokenizer.networks.configs")
            cv_module = importlib.import_module(
                "cosmos_tokenizer.networks.continuous_video"
            )
            return configs.continuous_video, cv_module.CausalContinuousVideoTokenizer
        finally:
            for k in list(sys.modules):
                if k == "cosmos_tokenizer" or k.startswith("cosmos_tokenizer."):
                    sys.modules.pop(k, None)
            sys.modules.update(saved_modules)
            sys.path = saved_path

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cosmos Tokenizer autoencoder model."""
        continuous_video_cfg, CVTokenizer = self._import_cosmos_package()

        config = continuous_video_cfg.copy()
        model = CVTokenizer(**config)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return synthetic video inputs for the Cosmos Tokenizer.

        The model expects input shape [B, C, T, H, W] with T as a multiple of 8 + 1.
        """
        return torch.randn(batch_size, 3, 9, 256, 256, dtype=torch.float32)
