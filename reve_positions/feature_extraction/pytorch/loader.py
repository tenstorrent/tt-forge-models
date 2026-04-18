# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
REVE Positions model loader for EEG electrode position feature extraction.
"""
from typing import Optional

import torch

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class RevePositionBankWrapper(torch.nn.Module):
    """Wrapper that takes one-hot encoded indices for XLA traceability."""

    def __init__(self, embedding):
        super().__init__()
        self.register_buffer("weight", embedding)

    def forward(self, one_hot: torch.Tensor):
        return one_hot @ self.weight


class ModelVariant(StrEnum):
    """Available REVE Positions model variants."""

    REVE_POSITIONS = "brain-bzh/reve-positions"


class ModelLoader(ForgeModel):
    """REVE Positions model loader for EEG electrode position feature extraction."""

    _VARIANTS = {
        ModelVariant.REVE_POSITIONS: ModelConfig(
            pretrained_model_name="brain-bzh/reve-positions",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.REVE_POSITIONS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._original_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="REVE Positions",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        original_model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        original_model.eval()
        self._original_model = original_model

        wrapper = RevePositionBankWrapper(original_model.embedding)
        wrapper.eval()
        self.model = wrapper

        return wrapper

    def load_inputs(self, dtype_override=None):
        electrode_names = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
        ]
        indices = [
            self._original_model.mapping[name]
            for name in electrode_names
            if name in self._original_model.mapping
        ]
        num_positions = len(self._original_model.position_names)
        one_hot = torch.zeros(len(indices), num_positions)
        for i, idx in enumerate(indices):
            one_hot[i, idx] = 1.0
        return (one_hot,)

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
