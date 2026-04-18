# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SONICS SpecTTTra model loader implementation for audio classification (synthetic song detection).
"""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SONICS SpecTTTra audio classification model variants."""

    ALPHA_120S = "Alpha_120s"


class ModelLoader(ForgeModel):
    """SONICS SpecTTTra model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.ALPHA_120S: ModelConfig(
            pretrained_model_name="awsaf49/sonics-spectttra-alpha-120s",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALPHA_120S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._full_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SpecTTTra",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sonics import HFAudioClassifier
        from sonics.models.model import use_global_pool

        full_model = HFAudioClassifier.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        full_model.eval()
        self._full_model = full_model

        class SpecTTTraEncoder(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.input_shape = model.input_shape
                self.encoder = model.encoder
                self.classifier = model.classifier
                self.model_name = model.model_name

            def forward(self, spec):
                spec = spec.unsqueeze(1)
                spec = F.interpolate(
                    spec, size=tuple(self.input_shape), mode="bilinear"
                )
                features = self.encoder(spec)
                embeds = (
                    features.mean(dim=1)
                    if use_global_pool(self.model_name)
                    else features
                )
                preds = self.classifier(embeds)
                return preds

        encoder = SpecTTTraEncoder(full_model)
        encoder.eval()
        if dtype_override is not None:
            encoder.to(dtype_override)

        return encoder

    def load_inputs(self, dtype_override=None):
        import torch

        sampling_rate = 16000
        duration_seconds = 1
        waveform = torch.randn(1, sampling_rate * duration_seconds)

        with torch.no_grad():
            spec = self._full_model.ft_extractor(waveform)

        if dtype_override is not None:
            spec = spec.to(dtype_override)

        return {"spec": spec}
