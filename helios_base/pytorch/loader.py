# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helios-Base text-to-video model loader implementation.

Helios-Base is a 14B parameter real-time long video generation model fine-tuned
from Wan2.1-T2V-14B. It uses a pyramid-based autoregressive approach generating
33 frames per chunk with v-prediction and a custom HeliosScheduler.
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline  # type: ignore[import]

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
    """Available Helios-Base model variants."""

    HELIOS_BASE = "Base"


class ModelLoader(ForgeModel):
    """Helios-Base text-to-video model loader implementation."""

    _VARIANTS = {
        ModelVariant.HELIOS_BASE: ModelConfig(
            pretrained_model_name="BestWishYsh/Helios-Base",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HELIOS_BASE

    DEFAULT_PROMPT = "A cat walks through a sunlit garden, soft lighting, cinematic, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Helios-Base",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self, dtype_override: Optional[torch.dtype] = None
    ) -> DiffusionPipeline:
        pipe_kwargs = {
            "torch_dtype": (
                dtype_override if dtype_override is not None else torch.bfloat16
            ),
        }

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        return self.pipeline

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the transformer from the Helios-Base pipeline.

        Args:
            dtype_override: Optional torch dtype to instantiate the pipeline with.

        Returns:
            torch.nn.Module: The HeliosTransformer3DModel used for denoising.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare synthetic inputs for the Helios-Base transformer model.

        Args:
            dtype_override: Optional torch dtype for the input tensors.

        Returns:
            dict: Keyword arguments for the transformer forward method.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self.load_model(dtype_override=dtype)

        config = self.pipeline.transformer.config

        batch_size = 1
        # Use small spatial dimensions for testing (64x64 pixels -> 8x8 latents)
        latent_h = 8
        latent_w = 8
        num_latent_frames = 1

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            num_latent_frames,
            latent_h,
            latent_w,
            dtype=dtype,
        )

        # max_sequence_length used by the Helios pipeline text encoder
        seq_len = 226
        encoder_hidden_states = torch.randn(
            batch_size, seq_len, config.text_dim, dtype=dtype
        )

        timestep = torch.tensor([500], dtype=torch.int64)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
