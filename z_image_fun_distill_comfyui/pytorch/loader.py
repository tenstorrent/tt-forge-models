# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UDCAI/Z-Image-Fun-Distill-ComfyUI model loader implementation.

ComfyUI-compatible conversion of the alibaba-pai/Z-Image-Fun-Lora-Distill LoRA
adapter for the Z-Image diffusion transformer. Selectively removes problematic
feed-forward keys while retaining the contrast/brightness improvements, enabling
stable 8-step text-to-image generation at CFG 1 across a variety of schedulers.

Available variants:
- DISTILL_8_STEPS_V1: 8-step ComfyUI-converted LoRA (v1, 201 MB)
- DISTILL_8_STEPS_2602: 8-step ComfyUI-converted LoRA (2602 variant, 320 MB)
"""

from typing import Optional, Dict, Any

import torch
from diffusers import DiffusionPipeline

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


BASE_MODEL_ID = "alibaba-pai/Z-Image"
LORA_REPO = "UDCAI/Z-Image-Fun-Distill-ComfyUI"


class ModelVariant(StrEnum):
    """Available UDCAI/Z-Image-Fun-Distill-ComfyUI model variants."""

    DISTILL_8_STEPS_V1 = "Distill_8_Steps_ComfyUI_v1"
    DISTILL_8_STEPS_2602 = "Distill_8_Steps_2602_UDCAI_ComfyUI"


_LORA_FILES = {
    ModelVariant.DISTILL_8_STEPS_V1: "Z-Image-Fun-Lora-Distill-8-Steps_ComfyUI_v1.safetensors",
    ModelVariant.DISTILL_8_STEPS_2602: "Z-Image-Fun-Lora-Distill-8-Steps-2602_UDCAI_ComfyUI.safetensors",
}


class ModelLoader(ForgeModel):
    """UDCAI/Z-Image-Fun-Distill-ComfyUI model loader implementation."""

    _VARIANTS = {
        ModelVariant.DISTILL_8_STEPS_V1: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
        ModelVariant.DISTILL_8_STEPS_2602: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DISTILL_8_STEPS_V1

    DEFAULT_PROMPT = "A serene mountain landscape at sunrise, photorealistic, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z-Image-Fun-Distill-ComfyUI",
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
        """Load Z-Image base pipeline and fuse ComfyUI-converted distill LoRA weights.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            DiffusionPipeline: Pipeline with distill LoRA weights fused.
        """
        pipe_dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=pipe_dtype,
            trust_remote_code=True,
        )

        weight_name = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=weight_name,
        )
        self.pipeline.fuse_lora(lora_scale=0.8)

        self.pipeline.to("cpu")

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load and return the Z-Image pipeline with ComfyUI distill LoRA.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            DiffusionPipeline: The Z-Image pipeline with ComfyUI distill LoRA fused.
        """
        if self.pipeline is None:
            return self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Load and return sample inputs for the Z-Image model.

        Args:
            prompt: Optional text prompt. Defaults to DEFAULT_PROMPT.

        Returns:
            dict: Input kwargs for the pipeline.
        """
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {
            "prompt": prompt_value,
            "guidance_scale": 1.0,
            "num_inference_steps": 8,
        }
