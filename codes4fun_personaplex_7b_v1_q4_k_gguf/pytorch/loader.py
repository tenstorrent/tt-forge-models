# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Codes4Fun PersonaPlex 7B v1 q4_k GGUF speech-text dialogue model loader implementation.
"""

import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
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
    """Available Codes4Fun PersonaPlex 7B v1 q4_k GGUF model variants."""

    PERSONAPLEX_7B_V1_Q4_K_GGUF = "PersonaPlex 7B v1 q4_k GGUF"


class ModelLoader(ForgeModel):
    """Codes4Fun PersonaPlex 7B v1 q4_k GGUF speech-text dialogue model loader implementation."""

    _VARIANTS = {
        ModelVariant.PERSONAPLEX_7B_V1_Q4_K_GGUF: ModelConfig(
            pretrained_model_name="Codes4Fun/personaplex-7b-v1-q4_k-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PERSONAPLEX_7B_V1_Q4_K_GGUF

    GGUF_FILE = "model-q4_k.gguf"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Codes4Fun PersonaPlex 7B v1 q4_k GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PersonaPlex LM model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        weight_path = hf_hub_download(pretrained_model_name, self.GGUF_FILE)

        dtype = dtype_override if dtype_override is not None else torch.float32
        model = loaders.get_moshi_lm(weight_path, device="cpu", dtype=dtype)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample code inputs for the PersonaPlex model.

        The model expects discrete audio codes of shape [B, K, T] where
        K=17 codebooks (1 text + 8 user audio + 8 agent audio) and T is time steps.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        weight_path = hf_hub_download(pretrained_model_name, self.GGUF_FILE)
        model = loaders.get_moshi_lm(weight_path, device="cpu", dtype=torch.float32)

        codes = torch.randint(0, model.card, (1, model.num_codebooks, 10))

        del model
        return codes
