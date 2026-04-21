# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HuBERT-ECG model loader implementation for ECG signal feature extraction.
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
    """Available HuBERT-ECG feature extraction model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """HuBERT-ECG model loader implementation for ECG signal feature extraction."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Edoardo-BS/hubert-ecg-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HuBERT-ECG",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import torch

        # Synthetic 12-lead ECG: 5 seconds at 100 Hz per lead, 12 leads
        # concatenated along the time axis to form a 1D input sequence.
        num_leads = 12
        samples_per_lead = 500
        input_values = torch.randn(1, num_leads * samples_per_lead, dtype=torch.float32)

        if dtype_override is not None:
            input_values = input_values.to(dtype_override)

        return {"input_values": input_values}
