# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Fun-Lora-Distill model loader implementation.

Distill LoRA adapter for the Z-Image diffusion transformer that distills both
inference steps and classifier-free guidance (CFG) for fast text-to-image
generation. Trained from scratch (not based on Z-Image-Turbo weights).

Available variants:
- DISTILL_8_STEPS: alibaba-pai/Z-Image-Fun-Lora-Distill 8-step distilled LoRA
"""

from typing import Optional, Any

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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


class ModelVariant(StrEnum):
    """Available Z-Image-Fun-Lora-Distill model variants."""

    DISTILL_8_STEPS = "Distill_8_Steps"


class ModelLoader(ForgeModel):
    """Z-Image-Fun-Lora-Distill model loader implementation."""

    _VARIANTS = {
        ModelVariant.DISTILL_8_STEPS: ModelConfig(
            pretrained_model_name="alibaba-pai/Z-Image-Fun-Lora-Distill",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DISTILL_8_STEPS

    DEFAULT_PROMPT = "A serene mountain landscape at sunrise, photorealistic, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z-Image-Fun-Lora-Distill",
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
        """Load Z-Image base pipeline and fuse distill LoRA weights.

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

        # Load LoRA weights with key fix for double-underscore prefix bug
        adapter_id = self._variant_config.pretrained_model_name
        weight_name = "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors"
        lora_path = hf_hub_download(adapter_id, weight_name)
        lora_state_dict = load_file(lora_path)
        lora_state_dict = {
            k.replace("lora_unet__", "lora_unet_"): v
            for k, v in lora_state_dict.items()
        }
        self.pipeline.load_lora_weights(lora_state_dict)
        self.pipeline.fuse_lora(lora_scale=0.8)

        self.pipeline.to("cpu")

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ) -> torch.nn.Module:
        """Load and return the Z-Image transformer with distill LoRA fused.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Z-Image transformer with distill LoRA fused.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Load and return sample inputs for the Z-Image transformer.

        Returns:
            list: Positional args [latent_input_list, timestep, prompt_embeds]
                  matching the transformer's forward(x, t, cap_feats) signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = self.DEFAULT_PROMPT

        if self.pipeline is None:
            self._load_pipeline(dtype)

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompt,
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
