#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AnimateDiff Motion LoRA Pan Left model loader implementation.

Loads the AnimateDiff pipeline with the Stable Diffusion v1.5 base model,
applies the motion adapter (guoyww/animatediff-motion-adapter-v1-5-2),
and loads the pan-left motion LoRA weights from
guoyww/animatediff-motion-lora-pan-left for text-to-video generation
with leftward camera panning.
"""

from typing import Any, Optional

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter  # type: ignore[import]

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

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
LORA_REPO = "guoyww/animatediff-motion-lora-pan-left"


class ModelVariant(StrEnum):
    """Available AnimateDiff Motion LoRA Pan Left variants."""

    PAN_LEFT = "PanLeft"


class ModelLoader(ForgeModel):
    """AnimateDiff Motion LoRA Pan Left model loader."""

    _VARIANTS = {
        ModelVariant.PAN_LEFT: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PAN_LEFT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[AnimateDiffPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ANIMATEDIFF_MOTION_LORA_PAN_LEFT",
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
        """Load the AnimateDiff pipeline with motion adapter and pan-left LoRA.

        Returns:
            UNetMotionModel extracted from the pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        adapter = MotionAdapter.from_pretrained(
            MOTION_ADAPTER,
            torch_dtype=dtype,
        )

        self.pipeline = AnimateDiffPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            motion_adapter=adapter,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        return self.pipeline.unet

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare tensor inputs for the AnimateDiff UNet.

        Returns:
            List of [sample, timestep, encoder_hidden_states] tensors.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype
        unet_cfg = self.pipeline.unet.config

        num_frames = 16
        sample_size = unet_cfg.sample_size
        in_channels = unet_cfg.in_channels
        cross_attention_dim = unet_cfg.cross_attention_dim

        prompt = (
            "A scenic mountain landscape with clouds drifting, "
            "cinematic pan left, smooth camera motion"
        )
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(text_inputs.input_ids)[
                0
            ].to(dtype)

        # UNet internally reshapes sample to (batch*num_frames, channels, h, w),
        # so encoder_hidden_states must match that batch dimension.
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            num_frames, dim=0
        )

        # UNetMotionModel.forward internally uses sample.shape[2] as num_frames,
        # so the expected shape is (batch, channels, num_frames, height, width).
        sample = torch.randn(
            1, in_channels, num_frames, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype)

        return [sample, timestep, encoder_hidden_states]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack UNet output to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
