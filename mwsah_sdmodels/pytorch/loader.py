# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MWSAH/sdmodels GGUF (MWSAH/sdmodels) model loader implementation.

HyperFlux Dedistilled is a 12B parameter text-to-image generation model in GGUF quantized format,
based on the FLUX transformer architecture.

Available variants:
- HYPERFLUX_DEDISTILLED_Q4_K_M: Q4_K_M quantized variant (~6.91 GB)
"""

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
from .src.model_utils import load_sdmodels_gguf_transformer, sdmodels_generate_inputs

REPO_ID = "MWSAH/sdmodels"


class ModelVariant(StrEnum):
    """Available MWSAH/sdmodels GGUF model variants."""

    HYPERFLUX_DEDISTILLED_Q4_K_M = "hyperFluxDedistilled_hyper16Q4KM"


class ModelLoader(ForgeModel):
    """MWSAH/sdmodels GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.HYPERFLUX_DEDISTILLED_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HYPERFLUX_DEDISTILLED_Q4_K_M

    GGUF_FILE = "hyperFluxDedistilled_hyper16Q4KM.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MWSAH/sdmodels GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._transformer is None:
            self._transformer = load_sdmodels_gguf_transformer(
                REPO_ID, self.GGUF_FILE, dtype=dtype_override
            )

        return self._transformer

    def load_inputs(self, dtype_override=None):
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        return sdmodels_generate_inputs(self._transformer, dtype=dtype_override)
