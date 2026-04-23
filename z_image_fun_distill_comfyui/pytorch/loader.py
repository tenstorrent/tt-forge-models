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

from typing import Optional

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
    ) -> None:
        """Load Z-Image base pipeline and fuse ComfyUI-converted distill LoRA weights."""
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

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load Z-Image pipeline and return the transformer submodule.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            ZImageTransformer2DModel: The core transformer module.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)
        elif dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline.transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ) -> list:
        """Prepare inputs for the Z-Image transformer's forward method.

        Args:
            dtype_override: Optional torch.dtype override.

        Returns:
            list: [latent_list, timestep, prompt_embeds] matching transformer forward(x, t, cap_feats).
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        transformer = self.pipeline.transformer
        dtype = next(transformer.parameters()).dtype

        # Encode text prompt into variable-length embeddings (list of tensors)
        with torch.no_grad():
            prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=self.DEFAULT_PROMPT,
                device="cpu",
                do_classifier_free_guidance=False,
            )

        # Build random latent in pipeline-expected shape
        in_channels = transformer.in_channels
        vae_scale_factor = self.pipeline.vae_scale_factor
        height, width = 1024, 1024
        h = 2 * (height // (vae_scale_factor * 2))
        w = 2 * (width // (vae_scale_factor * 2))

        latent = torch.randn(1, in_channels, h, w, dtype=dtype)
        # Pipeline unbinds along batch dim after adding temporal dim
        latent_list = list(latent.unsqueeze(2).unbind(dim=0))

        # Normalized timestep in [0, 1] (pipeline computes (1000 - t) / 1000)
        timestep = torch.tensor([0.5], dtype=dtype)

        if dtype_override is not None:
            latent_list = [t.to(dtype_override) for t in latent_list]
            timestep = timestep.to(dtype_override)
            prompt_embeds = [pe.to(dtype_override) for pe in prompt_embeds]

        return [latent_list, timestep, prompt_embeds]
