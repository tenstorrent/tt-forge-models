# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helios-Distilled-int8 model loader implementation for text-to-image generation.
"""

from typing import Optional, Dict, Any

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
        from diffusers import HeliosTransformer3DModel
        from transformers import UMT5Config, UMT5EncoderModel

        if extra_pipe_kwargs is None:
            extra_pipe_kwargs = {}

        torch_dtype = dtype_override if dtype_override is not None else torch.float32

        # The int8 model repo stores text_encoder and transformer weights as
        # root-level safetensors without per-component config.json files.
        # Pre-construct these so the remaining components load normally.
        text_encoder = UMT5EncoderModel(
            UMT5Config(d_model=4096, d_ff=10240, num_heads=64, d_kv=64, num_layers=2)
        )
        transformer = HeliosTransformer3DModel(num_layers=2)

        pipe_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "text_encoder": text_encoder,
            "transformer": transformer,
        }
        pipe_kwargs.update(extra_pipe_kwargs)

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if self.pipeline is None:
            self._load_pipeline(
                dtype_override=dtype_override,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                extra_pipe_kwargs=extra_pipe_kwargs,
            )

        transformer = self.pipeline.transformer
        transformer.eval()

        if dtype_override is not None:
            transformer = transformer.to(dtype=dtype_override)

        return transformer

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Dict[str, Any]:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        batch_size = 1
        in_channels = 16
        text_dim = 4096
        txt_seq_len = 32

        hidden_states = torch.randn(batch_size, in_channels, 1, 8, 8, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        timestep = torch.tensor([500], dtype=torch.long)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
