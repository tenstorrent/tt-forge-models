#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan diffusion model loader implementation.

Supports:
- Full pipeline loading (subfolder=None)
- VAE component loading (subfolder="vae") for encoder/decoder testing

Available variants:
- WAN22_TI2V_5B: Wan 2.2 text-to-image-to-video 5B (full pipeline only)
- WAN21_T2V_14B: Wan 2.1 text-to-video 14B (supports VAE subfolder)
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
    load_vae,
    load_vae_decoder_inputs,
    load_vae_encoder_inputs,
)

# Supported subfolders for loading individual components
SUPPORTED_SUBFOLDERS = {"vae"}


class ModelVariant(StrEnum):
    """Available Wan diffusion model variants."""

    WAN22_TI2V_5B = "2.2_Ti2v_5B"
    WAN21_T2V_14B = "2.1_T2v_14B"


class ModelLoader(ForgeModel):
    """
    Loader for Wan diffusion models.

    Supports loading the full pipeline or specific components via subfolder:
    - subfolder=None: Load full DiffusionPipeline
    - subfolder="vae": Load AutoencoderKLWan (~508MB)
    """

    _VARIANTS = {
        ModelVariant.WAN22_TI2V_5B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ),
        ModelVariant.WAN21_T2V_14B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_TI2V_5B

    # Reuse the prompt from the reference inference script for smoke testing.
    DEFAULT_PROMPT = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: Optional[str] = None,
    ):
        """
        Initialize the model loader.

        Args:
            variant: Model variant to load
            subfolder: Optional subfolder to load specific component:
                - None: Load full DiffusionPipeline
                - 'vae': Load AutoencoderKLWan
        """
        super().__init__(variant)
        if subfolder is not None and subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

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
        Load and return the model or component.

        When subfolder is None, loads the full DiffusionPipeline.
        When subfolder is "vae", loads AutoencoderKLWan.
        """
        config = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self._subfolder == "vae":
            return load_vae(config.pretrained_model_name, dtype)

        # Full pipeline loading
        if extra_pipe_kwargs is None:
            extra_pipe_kwargs = {}

        pipe_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        pipe_kwargs.update(extra_pipe_kwargs)

        pipeline = DiffusionPipeline.from_pretrained(
            config.pretrained_model_name,
            **pipe_kwargs,
        )

        return pipeline

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        """
        Load sample inputs for the model or component.

        For VAE subfolder, pass vae_type="decoder" or vae_type="encoder".
        For full pipeline, returns a prompt dict.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

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

        # Full pipeline inputs
        prompt = kwargs.get("prompt", self.DEFAULT_PROMPT)
        return {"prompt": prompt}

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        """Unpack model output to extract tensor."""
        if hasattr(output, "sample"):
            return output.sample
        elif isinstance(output, tuple):
            return output[0]
        return output
