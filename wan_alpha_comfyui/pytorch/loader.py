#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan-Alpha ComfyUI model loader implementation.

Loads the Wan 2.1 T2V base pipeline and applies the RGBA LoRA weights from
htdong/Wan-Alpha_ComfyUI for text-to-video generation with alpha-channel
(transparent-background) output.

The transformer (WanTransformer3DModel) is extracted from the pipeline and
returned as the model for compilation. LoRA weights are applied before
extraction when possible; if the diffusers conversion for this LoRA format
fails, LoRA loading is skipped gracefully so the base model can be compiled.

Repository: https://huggingface.co/htdong/Wan-Alpha_ComfyUI
"""

import warnings
from typing import Any, Optional

import torch
from diffusers import WanPipeline  # type: ignore[import]

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

BASE_MODEL = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
LORA_REPO = "htdong/Wan-Alpha_ComfyUI"
LORA_FILENAME = "epoch-13-1500_changed.safetensors"

# Transformer input dimensions for Wan2.1-T2V-14B
_TRANSFORMER_IN_CHANNELS = 16
_LATENT_HEIGHT = 4
_LATENT_WIDTH = 4
_LATENT_DEPTH = 2
_TEXT_HIDDEN_DIM = 4096
_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan-Alpha ComfyUI model variants."""

    WAN_ALPHA = "Wan-Alpha"


class ModelLoader(ForgeModel):
    """Wan-Alpha ComfyUI model loader for RGBA text-to-video generation."""

    _VARIANTS = {
        ModelVariant.WAN_ALPHA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN_ALPHA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_ALPHA_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Wan 2.1 T2V transformer with Wan-Alpha LoRA weights applied.

        Loads the full pipeline, applies LoRA weights where possible, then
        extracts and returns the transformer (WanTransformer3DModel) so that
        the test framework receives a torch.nn.Module.

        Returns:
            WanTransformer3DModel with LoRA weights fused when loadable.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = WanPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        try:
            self.pipeline.load_lora_weights(
                LORA_REPO,
                weight_name=LORA_FILENAME,
            )
        except (IndexError, ValueError) as e:
            # The Wan-Alpha LoRA ships in ComfyUI-native format which diffusers
            # may not yet convert cleanly onto the standard T2V base model.
            # Skip LoRA loading so the base pipeline can still be compiled.
            warnings.warn(
                f"Skipping LoRA loading for {LORA_FILENAME}: {e}",
                stacklevel=2,
            )

        transformer = self.pipeline.transformer
        transformer.eval()
        return transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare synthetic inputs for the WanTransformer3DModel forward pass.

        Returns:
            dict with hidden_states, encoder_hidden_states, timestep, and
            return_dict keys suitable for WanTransformer3DModel.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = 1

        hidden_states = torch.randn(
            batch_size,
            _TRANSFORMER_IN_CHANNELS,
            _LATENT_DEPTH,
            _LATENT_HEIGHT,
            _LATENT_WIDTH,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size, _TEXT_SEQ_LEN, _TEXT_HIDDEN_DIM, dtype=dtype
        )
        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
