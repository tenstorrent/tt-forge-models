# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helios-Distilled-int8 model loader implementation for text-to-image generation.
"""

from typing import Optional, Dict, Any

import torch
from diffusers import DiffusionPipeline, HeliosTransformer3DModel
from transformers import AutoConfig, UMT5EncoderModel

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

# The reference model provides the text_encoder config, since the int8 repo
# only has flat safetensors at root without subdirectory configs.
_REFERENCE_MODEL = "BestWishYsh/Helios-Distilled"


class ModelVariant(StrEnum):
    """Available Helios-Distilled-int8 model variants."""

    HELIOS_DISTILLED_INT8 = "Helios-Distilled-int8"


class ModelLoader(ForgeModel):
    """Helios-Distilled-int8 model loader implementation."""

    _VARIANTS = {
        ModelVariant.HELIOS_DISTILLED_INT8: ModelConfig(
            pretrained_model_name="szwagros/Helios-Distilled-int8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HELIOS_DISTILLED_INT8

    DEFAULT_PROMPT = "A cinematic portrait of a robot in a neon-lit lab"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Helios-Distilled",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DiffusionPipeline:
        if extra_pipe_kwargs is None:
            extra_pipe_kwargs = {}

        # Use bfloat16 as default; the model is ~26B params and float32 needs ~104 GB RAM.
        model_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # The int8 repo (szwagros/Helios-Distilled-int8) omits text_encoder/ and
        # transformer/ subdirectories, so DiffusionPipeline.from_pretrained fails
        # looking for config.json. Pre-initialize these components from the reference
        # model's configs (small downloads) so from_pretrained loads scheduler/
        # tokenizer/vae from the int8 repo without error.
        transformer = HeliosTransformer3DModel().to(dtype=model_dtype)

        text_encoder_config = AutoConfig.from_pretrained(
            _REFERENCE_MODEL, subfolder="text_encoder"
        )
        text_encoder = UMT5EncoderModel(text_encoder_config).to(dtype=model_dtype)

        pipe_kwargs = {
            "torch_dtype": model_dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "transformer": transformer,
            "text_encoder": text_encoder,
        }
        pipe_kwargs.update(extra_pipe_kwargs)

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Load and return the Helios transformer denoiser from the pipeline.
        """
        if self.pipeline is None:
            self._load_pipeline(
                dtype_override=dtype_override,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                extra_pipe_kwargs=extra_pipe_kwargs,
            )

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(
                dtype=dtype_override
            )

        return self.pipeline.transformer

    def load_inputs(
        self, prompt: Optional[str] = None, batch_size: int = 1
    ) -> Dict[str, Any]:
        """Return dummy inputs for the HeliosTransformer3DModel forward pass."""
        # Determine dtype from the loaded transformer, defaulting to bfloat16
        if self.pipeline is not None:
            model_dtype = next(self.pipeline.transformer.parameters()).dtype
        else:
            model_dtype = torch.bfloat16

        # hidden_states: (B, in_channels=16, frames, H, W) — VAE latents before denoising
        hidden_states = torch.zeros(batch_size, 16, 1, 8, 8, dtype=model_dtype)

        # timestep: (B,) — denoising step index
        timestep = torch.tensor([500] * batch_size, dtype=torch.long)

        # encoder_hidden_states: (B, seq_len, text_dim=4096) — text encoder output
        encoder_hidden_states = torch.zeros(batch_size, 16, 4096, dtype=model_dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
