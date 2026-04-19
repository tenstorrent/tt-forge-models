# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image Turbo Training Adapter (ostris/zimage_turbo_training_adapter) model loader.

This is a LoRA de-distillation adapter for the Tongyi-MAI/Z-Image-Turbo base model.
It is designed to be stacked during fine-tuning to preserve step-distillation speed.
The adapter loads the base Z-Image-Turbo pipeline and applies the LoRA weights.

Available variants:
- ZIMAGE_TURBO_TRAINING_ADAPTER_V1: ostris/zimage_turbo_training_adapter (v1 weights)
"""

from typing import Any, Optional

import torch
from diffusers import AutoPipelineForText2Image

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


BASE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"
ADAPTER_REPO_ID = "ostris/zimage_turbo_training_adapter"


class ModelVariant(StrEnum):
    """Available Z-Image Turbo Training Adapter model variants."""

    ZIMAGE_TURBO_TRAINING_ADAPTER_V1 = "ZImage_Turbo_Training_Adapter_v1"


class ModelLoader(ForgeModel):
    """Z-Image Turbo Training Adapter model loader implementation."""

    _VARIANTS = {
        ModelVariant.ZIMAGE_TURBO_TRAINING_ADAPTER_V1: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ZIMAGE_TURBO_TRAINING_ADAPTER_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ZImage_Turbo_Training_Adapter",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Z-Image-Turbo base pipeline and apply the LoRA adapter weights.

        Returns:
            AutoPipelineForText2Image: The pipeline with LoRA adapter applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        lora_path = hf_hub_download(
            ADAPTER_REPO_ID, "zimage_turbo_training_adapter_v1.safetensors"
        )
        state_dict = load_file(lora_path)

        # LoRA weights lack alpha keys; add default alpha=rank so scaling is 1.0
        for key in list(state_dict.keys()):
            if key.endswith(".lora_A.weight"):
                alpha_key = key.replace(".lora_A.weight", ".alpha")
                if alpha_key not in state_dict:
                    rank = state_dict[key].shape[0]
                    state_dict[alpha_key] = torch.tensor(float(rank))

        self.pipeline.load_lora_weights(state_dict)
        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> Any:
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."

        if self.pipeline is None:
            self.load_model(dtype_override=dtype)

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
