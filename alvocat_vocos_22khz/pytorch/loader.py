# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
alVoCat Vocos 22kHz model loader implementation.

alVoCat is a Vocos-based neural vocoder for Catalan text-to-speech that
synthesizes 22 kHz audio waveforms from 80-bin mel-spectrograms.
"""
import importlib

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


def _instantiate_class(init):
    """Instantiate a class from a config dict with class_path and init_args."""
    kwargs = init.get("init_args", {})
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = importlib.import_module(class_module)
    cls = getattr(module, class_name)
    return cls(**kwargs)


class VocosDecodeWrapper(nn.Module):
    """Wrapper that exposes vocos backbone + head as a single forward pass.

    Accepts pre-computed mel features directly, bypassing the feature extractor.
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, features):
        x = self.backbone(features)
        audio = self.head(x)
        return audio


class ModelVariant(StrEnum):
    """Available alVoCat Vocos model variants."""

    ALVOCAT_22KHZ = "alvocat_22khz"


class ModelLoader(ForgeModel):
    """alVoCat Vocos 22kHz model loader implementation."""

    _VARIANTS = {
        ModelVariant.ALVOCAT_22KHZ: ModelConfig(
            pretrained_model_name="projecte-aina/alvocat-vocos-22khz",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALVOCAT_22KHZ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="AlVoCatVocos",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the alVoCat Vocos model wrapped for decode inference.

        Loads backbone and head directly from the HuggingFace config to avoid
        importing vocos.feature_extractors, which has a top-level dependency on
        the 'encodec' PyPI package that conflicts with the local encodec/ model
        directory present in this repo.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: Wrapped Vocos model that decodes mel features to audio.
        """
        import sys
        import yaml
        from huggingface_hub import hf_hub_download

        # Pre-import encodec from site-packages before vocos does it, so
        # sys.modules caches the real PyPI package instead of the local
        # encodec/ model directory (which lacks EncodecModel).
        if "encodec" not in sys.modules:
            site_pkgs = [p for p in sys.path if "site-packages" in p]
            other = [p for p in sys.path if "site-packages" not in p]
            sys.path[:] = site_pkgs + other
            try:
                import encodec  # noqa: F401
            finally:
                sys.path[:] = other + site_pkgs

        pretrained_model_name = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(
            repo_id=pretrained_model_name, filename="config.yaml"
        )
        model_path = hf_hub_download(
            repo_id=pretrained_model_name, filename="pytorch_model.bin"
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        backbone = _instantiate_class(config["backbone"])
        head = _instantiate_class(config["head"])

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        backbone_sd = {
            k[len("backbone.") :]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        head_sd = {
            k[len("head.") :]: v for k, v in state_dict.items() if k.startswith("head.")
        }
        backbone.load_state_dict(backbone_sd)
        head.load_state_dict(head_sd)

        model = VocosDecodeWrapper(backbone, head)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the alVoCat Vocos model.

        Generates a random 80-bin mel-spectrogram shaped (batch, n_mels, frames)
        suitable for the Vocos decode backbone + head.

        Args:
            dtype_override: Optional torch.dtype to override the input tensor's default dtype.

        Returns:
            dict: Input tensors containing the mel-spectrogram features.
        """
        # Shape: (batch, n_mels=80, frames)
        features = torch.randn(1, 80, 256)

        if dtype_override is not None:
            features = features.to(dtype_override)

        return {"features": features}
