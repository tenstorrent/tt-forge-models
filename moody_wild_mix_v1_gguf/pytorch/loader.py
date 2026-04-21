# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moody Wild Mix v1 GGUF (Gthalmie1/moody-wild-mix-v1-gguf) model loader implementation.

Moody Wild Mix v1 is a text-to-image generation model in GGUF quantized format,
based on the Lumina-Image-2.0 architecture (~6B parameters). The original weights
come from the CivitAI "Moody Wild Mix" model by catlover1937.

Available variants:
- BASE_Q4_K_M: Base 50-step Q4_K_M quantized transformer
- DISTILLED_Q4_K_M: Distilled 10-step Q4_K_M quantized transformer
"""

from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download

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

REPO_ID = "Gthalmie1/moody-wild-mix-v1-gguf"

# Lumina-Image-2.0 architecture constants
IN_CHANNELS = 16
CAP_FEAT_DIM = 2304  # Gemma-2-2b hidden size


class ModelVariant(StrEnum):
    """Available Moody Wild Mix v1 GGUF model variants."""

    BASE_Q4_K_M = "moodyWildMix_v10Base50steps_Q4_K_M"
    DISTILLED_Q4_K_M = "moodyWildMix_v10Distilled10steps_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.BASE_Q4_K_M: "moodyWildMix_v10Base50steps_q4_k_m.gguf",
    ModelVariant.DISTILLED_Q4_K_M: "moodyWildMix_v10Distilled10steps_q4_k_m.gguf",
}


class ModelLoader(ForgeModel):
    """Moody Wild Mix v1 GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.BASE_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.DISTILLED_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Moody_Wild_Mix_v1_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the GGUF-quantized Lumina2 transformer."""
        from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=gguf_filename,
        )

        self._transformer = Lumina2Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1) -> Any:
        """Prepare synthetic transformer inputs for the Moody Wild Mix v1 GGUF model."""
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Latent image: (B, in_channels, H, W)
        height = 128
        width = 128
        hidden_states = torch.randn(batch_size, IN_CHANNELS, height, width, dtype=dtype)

        # Timestep: (B,)
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        # Text encoder hidden states from Gemma-2-2b: (B, seq_len, cap_feat_dim)
        max_sequence_length = 128
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, CAP_FEAT_DIM, dtype=dtype
        )

        # Encoder attention mask: (B, seq_len)
        encoder_attention_mask = torch.ones(
            batch_size, max_sequence_length, dtype=torch.bool
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
