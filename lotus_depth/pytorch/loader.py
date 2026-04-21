# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lotus Depth model loader implementation for diffusion-based monocular depth estimation.

Loads the UNet from the Lotus-D depth estimation pipeline and pre-computes
latent inputs, since LotusDPipeline is not available in the diffusers package.
"""
from typing import Any, Optional

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

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


class ModelVariant(StrEnum):
    """Available Lotus Depth model variants."""

    DEPTH_D_V1_1 = "Depth-D-v1-1"


class ModelLoader(ForgeModel):
    """Lotus Depth model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEPTH_D_V1_1: ModelConfig(
            pretrained_model_name="jingheya/lotus-depth-d-v1-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEPTH_D_V1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LotusDepth",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override or torch.float32

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name, subfolder="unet", torch_dtype=dtype
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name, subfolder="text_encoder", torch_dtype=dtype
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name, subfolder="tokenizer"
        )

        return self.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.unet is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.unet.dtype
        in_channels = self.unet.config.in_channels
        sample_size = self.unet.config.sample_size

        text_inputs = self.tokenizer(
            "depth",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(text_inputs.input_ids)[0].to(
                dtype
            )

        latent_sample = torch.randn(
            batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([999], dtype=dtype)
        class_labels = torch.zeros(batch_size, 4, dtype=dtype)

        return {
            "sample": latent_sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "class_labels": class_labels,
        }

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
