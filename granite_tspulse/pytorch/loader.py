# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite TSPulse model loader implementation for time series anomaly detection.
"""

import torch
from typing import Optional
from dataclasses import dataclass


def _patch_transformers_for_tsfm():
    """Patch transformers.utils with functions removed in v5 but needed by tsfm_public."""
    import transformers.utils as tu

    if not hasattr(tu, "is_remote_url"):
        from urllib.parse import urlparse

        def is_remote_url(url_or_filename):
            parsed = urlparse(str(url_or_filename))
            return parsed.scheme in ("http", "https")

        tu.is_remote_url = is_remote_url

    if not hasattr(tu, "is_offline_mode"):
        from huggingface_hub import is_offline_mode

        tu.is_offline_mode = is_offline_mode

    if not hasattr(tu, "download_url"):
        import tempfile
        import os
        import urllib.request

        def download_url(url, proxies=None):
            tmp_fd, tmp_file = tempfile.mkstemp()
            os.close(tmp_fd)
            urllib.request.urlretrieve(url, tmp_file)
            return tmp_file

        tu.download_url = download_url


_patch_transformers_for_tsfm()

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


@dataclass
class GraniteTSPulseConfig(ModelConfig):
    context_length: int = 512


class ModelVariant(StrEnum):
    R1 = "r1"


class ModelLoader(ForgeModel):
    """Granite TSPulse model loader for time series anomaly detection.

    Loads the IBM Granite TSPulse R1 model for zero-shot
    time series anomaly detection using dual-space masked reconstruction.
    """

    _VARIANTS = {
        ModelVariant.R1: GraniteTSPulseConfig(
            pretrained_model_name="ibm-granite/granite-timeseries-tspulse-r1",
            context_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.R1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Granite-TSPulse",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Granite TSPulse model for time series anomaly detection.

        Returns:
            torch.nn.Module: The TSPulse reconstruction model instance.
        """
        cfg = self._variant_config

        from tsfm_public.models.tspulse import TSPulseForReconstruction
        from tsfm_public.models.tspulse.configuration_tspulse import TSPulseConfig

        hf_config = TSPulseConfig.from_pretrained(cfg.pretrained_model_name)
        hf_config.post_init = True

        model = TSPulseForReconstruction.from_pretrained(
            cfg.pretrained_model_name, config=hf_config
        )

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'past_values' tensor of shape
                  (batch, context_length, num_channels).
        """
        cfg = self._variant_config
        dtype = torch.float32

        torch.manual_seed(42)
        # TSPulse expects (batch, context_length, num_input_channels)
        past_values = torch.randn(1, cfg.context_length, 1, dtype=dtype)

        return {"past_values": past_values}
