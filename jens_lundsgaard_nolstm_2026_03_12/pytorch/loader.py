# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ConvLSTM Autoencoder model loader for JensLundsgaard/nolstm-2026-03-12.

The HuggingFace model was published with PyTorchModelHubMixin and relies on
custom Python modules (raffael_model.py, raffael_conv_lstm.py), so it cannot
be loaded via transformers.AutoModel. We vendor the architecture locally and
load config + safetensors weights from the Hub directly.
"""
import json
from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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
from .src.model import ConvLSTMAutoencoder


@dataclass
class ConvLSTMAEConfig(ModelConfig):
    pass


class ModelVariant(StrEnum):
    NOLSTM_2026_03_12 = "nolstm-2026-03-12"


class ModelLoader(ForgeModel):
    """Loader for the JensLundsgaard/nolstm-2026-03-12 ConvLSTM autoencoder."""

    _VARIANTS = {
        ModelVariant.NOLSTM_2026_03_12: ConvLSTMAEConfig(
            pretrained_model_name="JensLundsgaard/nolstm-2026-03-12",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOLSTM_2026_03_12

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._hf_config: Optional[dict] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ConvLSTMAutoencoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_hf_config(self) -> dict:
        if self._hf_config is None:
            repo_id = self._variant_config.pretrained_model_name
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config_path) as f:
                self._hf_config = json.load(f)
        return self._hf_config

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the vendored ConvLSTMAutoencoder with pretrained safetensors weights."""
        repo_id = self._variant_config.pretrained_model_name
        config = self._load_hf_config()

        model = ConvLSTMAutoencoder(config=config)

        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, *, dtype_override=None, batch_size=1):
        """Return a synthetic (B, T, C, H, W) video clip matching the model config."""
        config = self._load_hf_config()
        seq_len = config.get("seq_len", 50)
        channels = config.get("input_channels", 1)
        image_size = config.get("image_size", 128)

        dtype = dtype_override or torch.float32
        inputs = torch.rand(
            batch_size, seq_len, channels, image_size, image_size, dtype=dtype
        )
        return inputs
