# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TimesFM 1.0 200M model loader implementation for time series forecasting.
"""

import os
from typing import Optional

import torch

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available TimesFM 1.0 model variants."""

    TIMESFM_1_0_200M = "TimesFM_1_0_200M"


class ModelLoader(ForgeModel):
    """TimesFM 1.0 model loader for time series forecasting."""

    _VARIANTS = {
        ModelVariant.TIMESFM_1_0_200M: ModelConfig(
            pretrained_model_name="google/timesfm-1.0-200m-pytorch",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TIMESFM_1_0_200M

    # TimesFM 1.0 200M model architecture parameters
    _MODEL_CONFIG = {
        "num_layers": 20,
        "num_heads": 16,
        "hidden_size": 1280,
        "intermediate_size": 1280,
        "patch_len": 32,
        "horizon_len": 128,
        "head_dim": 80,
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TimesFM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TimesFM 1.0 200M PyTorch model."""
        import sys

        src_dir = os.path.join(os.path.dirname(__file__), "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        import pytorch_patched_decoder as ppd
        from huggingface_hub import snapshot_download

        config = ppd.TimesFMConfig(**self._MODEL_CONFIG)
        model = ppd.PatchedTimeSeriesDecoder(config)

        repo_dir = snapshot_download(self._variant_config.pretrained_model_name)
        checkpoint_path = os.path.join(repo_dir, "torch_model.ckpt")
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load sample inputs for the TimesFM 1.0 model.

        Returns tensors matching PatchedTimeSeriesDecoder.forward(input_ts, input_padding, freq).
        """
        torch.manual_seed(42)
        batch_size = 1
        context_len = 512

        input_ts = torch.randn(batch_size, context_len)
        input_padding = torch.zeros(batch_size, context_len)
        freq = torch.zeros(batch_size, 1, dtype=torch.long)

        if dtype_override is not None:
            input_ts = input_ts.to(dtype_override)
            input_padding = input_padding.to(dtype_override)

        return {
            "input_ts": input_ts,
            "input_padding": input_padding,
            "freq": freq,
        }
