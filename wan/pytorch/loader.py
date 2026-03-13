#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan diffusion model loader implementation.

Supports:
- Full pipeline loading (subfolder=None)
- VAE component loading (subfolder="vae") for encoder/decoder testing
- Text encoder loading (subfolder="text_encoder") for UMT5 testing
- Transformer loading (subfolder="transformer") for WanTransformer3D testing

Available variants:
- WAN22_TI2V_5B: Wan 2.2 text-to-image-to-video 5B (full pipeline only)
- WAN21_T2V_14B: Wan 2.1 text-to-video 14B
- WAN21_T2V_13B: Wan 2.1 text-to-video 1.3B (lighter variant)
"""

from typing import Any, Optional, Dict

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
from .src.utils import (
    load_text_encoder,
    load_text_encoder_inputs,
    load_transformer,
    load_transformer_inputs,
    load_vae,
    load_vae_decoder_inputs,
    load_vae_encoder_inputs,
)

SUPPORTED_SUBFOLDERS = {"vae", "text_encoder", "transformer"}


class ModelVariant(StrEnum):
    """Available Wan diffusion model variants."""

    WAN22_TI2V_5B = "2.2_Ti2v_5B"
    WAN21_T2V_14B = "2.1_T2v_14B"
    WAN21_T2V_13B = "2.1_T2v_1.3B"


class ModelLoader(ForgeModel):
    """Wan diffusion model loader that mirrors the standalone inference script."""

    _VARIANTS = {
        ModelVariant.WAN22_TI2V_5B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ),
        ModelVariant.WAN21_T2V_14B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        ),
        ModelVariant.WAN21_T2V_13B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_TI2V_5B

    # Reuse the prompt from the reference inference script for smoke testing.
    DEFAULT_PROMPT = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, subfolder: Optional[str] = None
    ):
        super().__init__(variant)
        if subfolder is not None and subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VIDEO_TTT
            if variant in (ModelVariant.WAN21_T2V_14B, ModelVariant.WAN21_T2V_13B)
            else ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
        device_map: str = "cpu",
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DiffusionPipeline:
        if extra_pipe_kwargs is None:
            extra_pipe_kwargs = {}

        pipe_kwargs = {
            "torch_dtype": (
                dtype_override if dtype_override is not None else torch.float32
            ),
            "device_map": device_map,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        pipe_kwargs.update(extra_pipe_kwargs)

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        # Align dtype/device post creation in case caller wants something else
        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        device_map: str = "cpu",
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Load and return the Wan diffusion pipeline or an individual component.

        Args:
            dtype_override: Optional torch dtype to instantiate/convert the pipeline with.
            device_map: Device placement passed through to DiffusionPipeline.
            low_cpu_mem_usage: Whether to enable the huggingface low-memory loading path.
            extra_pipe_kwargs: Additional kwargs forwarded to DiffusionPipeline.from_pretrained.

        Returns:
            DiffusionPipeline, AutoencoderKLWan, UMT5EncoderModel, or
            WanTransformer3DModel depending on subfolder.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self._subfolder == "text_encoder":
            return load_text_encoder(
                self._variant_config.pretrained_model_name, dtype
            )

        if self._subfolder == "transformer":
            return load_transformer(
                self._variant_config.pretrained_model_name, dtype
            )

        if self._subfolder == "vae":
            return load_vae(self._variant_config.pretrained_model_name, dtype)

        if self.pipeline is None:
            return self._load_pipeline(
                dtype_override=dtype_override,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                extra_pipe_kwargs=extra_pipe_kwargs,
            )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """
        Prepare inputs for the model or component.

        For VAE subfolder, pass vae_type="decoder" or vae_type="encoder".
        For text_encoder/transformer subfolder, returns appropriate input dicts.
        For full pipeline, returns a prompt dict.
        """
        dtype = kwargs.get("dtype_override", torch.float32)

        if self._subfolder == "text_encoder":
            return load_text_encoder_inputs(dtype)

        if self._subfolder == "transformer":
            return load_transformer_inputs(dtype)

        if self._subfolder == "vae":
            vae_type = kwargs.get("vae_type")
            if vae_type == "decoder":
                return load_vae_decoder_inputs(dtype)
            elif vae_type == "encoder":
                return load_vae_encoder_inputs(dtype)
            else:
                raise ValueError(
                    f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
                )

        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
