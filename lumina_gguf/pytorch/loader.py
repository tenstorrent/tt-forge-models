# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
calcuis/lumina-gguf model loader implementation for text-to-image generation.

Loads the calcuis/lumina-gguf GGUF-quantized Lumina-Image-2.0 transformer using
diffusers' GGUF quantization support.

Repository:
- https://huggingface.co/calcuis/lumina-gguf
"""
import torch
from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel
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

REPO_ID = "calcuis/lumina-gguf"

# Lumina-Image-2.0 architecture constants
IN_CHANNELS = 16
CAP_FEAT_DIM = 2304  # Gemma-2-2b hidden size


class ModelVariant(StrEnum):
    """Available calcuis/lumina-gguf model variants."""

    Q4_K_S = "Q4_K_S"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """calcuis/lumina-gguf model loader implementation for text-to-image tasks."""

    _VARIANTS = {
        ModelVariant.Q4_K_S: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K_S: "lumina2-q4_k_s.gguf",
        ModelVariant.Q8_0: "lumina2-q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="calcuis/lumina-gguf",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        repo_id = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.transformer = Lumina2Transformer2DModel.from_single_file(
            f"https://huggingface.co/{repo_id}/blob/main/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )
        self.transformer.eval()

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
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

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }

        return inputs
