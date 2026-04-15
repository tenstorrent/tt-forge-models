# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boreal-Qwen-Image model loader implementation for text-to-image generation.

Loads LoRA adapter weights from kudzueye/boreal-qwen-image on top of the
Qwen/Qwen-Image base diffusion model. The transformer (QwenImageTransformer2DModel)
is extracted from the pipeline and returned as the model for compilation.

Available variants:
- BLEND_LOW_RANK: Boreal blend style (low rank)
- GENERAL_DISCRETE_LOW_RANK: General discrete style (low rank)
- PORTRAITS_HIGH_RANK: Portrait style (high rank)
- SMALL_DISCRETE_LOW_RANK: Small discrete style (low rank)
"""

from typing import Any, Dict, Optional

import torch
from diffusers import DiffusionPipeline

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

LORA_REPO_ID = "kudzueye/boreal-qwen-image"
BASE_MODEL_ID = "Qwen/Qwen-Image"

# Transformer input dimensions for QwenImageTransformer2DModel
_IN_CHANNELS = 64  # in_channels after packing (16 channels * 4 from 2x2 patches)
_JOINT_ATTENTION_DIM = 3584
_LATENT_H = 4  # Small latent height for testing
_LATENT_W = 4  # Small latent width for testing
_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Boreal-Qwen-Image LoRA variants."""

    BLEND_LOW_RANK = "blend-low-rank"
    GENERAL_DISCRETE_LOW_RANK = "general-discrete-low-rank"
    PORTRAITS_HIGH_RANK = "portraits-high-rank"
    SMALL_DISCRETE_LOW_RANK = "small-discrete-low-rank"


_LORA_FILES = {
    ModelVariant.BLEND_LOW_RANK: "qwen-boreal-blend-low-rank.safetensors",
    ModelVariant.GENERAL_DISCRETE_LOW_RANK: "qwen-boreal-general-discrete-low-rank.safetensors",
    ModelVariant.PORTRAITS_HIGH_RANK: "qwen-boreal-portraits-portraits-high-rank.safetensors",
    ModelVariant.SMALL_DISCRETE_LOW_RANK: "qwen-boreal-small-discrete-low-rank.safetensors",
}


class ModelLoader(ForgeModel):
    """Boreal-Qwen-Image LoRA model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.BLEND_LOW_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
        ModelVariant.GENERAL_DISCRETE_LOW_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
        ModelVariant.PORTRAITS_HIGH_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
        ModelVariant.SMALL_DISCRETE_LOW_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERAL_DISCRETE_LOW_RANK

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Boreal-Qwen-Image",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
    ) -> DiffusionPipeline:
        """Load the base Qwen-Image pipeline and apply LoRA weights."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO_ID,
            weight_name=_LORA_FILES[self._variant],
        )

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load and return the Boreal-Qwen-Image transformer with LoRA weights."""
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        transformer = self.pipeline.transformer

        if dtype_override is not None:
            transformer = transformer.to(dtype_override)

        return transformer

    def load_inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare synthetic inputs for the QwenImageTransformer2DModel forward pass."""
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = 1
        # After packing, seq_len = (latent_h // 2) * (latent_w // 2)
        seq_len = (_LATENT_H // 2) * (_LATENT_W // 2)

        hidden_states = torch.randn(batch_size, seq_len, _IN_CHANNELS, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, _TEXT_SEQ_LEN, _JOINT_ATTENTION_DIM, dtype=dtype
        )
        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        img_shapes = [[(_LATENT_H // 2, _LATENT_W // 2)]] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
