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
from diffusers import ZImagePipeline

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


BASE_MODEL_ID = "Tongyi-MAI/Z-Image"
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
        self.pipeline: Optional[ZImagePipeline] = None

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
    ) -> ZImagePipeline:
        """Load Z-Image base pipeline and fuse ComfyUI-converted distill LoRA weights."""
        pipe_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.pipeline = ZImagePipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=pipe_dtype,
            low_cpu_mem_usage=False,
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
        """Load and return the Z-Image transformer with ComfyUI distill LoRA fused."""
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)
        elif dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Load and return tensor inputs for the Z-Image transformer.

        Returns:
            list: [latent_input_list, timestep, prompt_embeds]
        """
        if self.pipeline is None:
            self._load_pipeline()

        dtype = self.pipeline.transformer.dtype
        height = 128
        width = 128
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompt_value,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        num_channels_latents = self.pipeline.transformer.in_channels
        vae_scale = self.pipeline.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack transformer output to sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
