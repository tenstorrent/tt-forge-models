# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional

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
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
from .src.model_utils import stable_diffusion_preprocessing_xl


LORA_REPO = "ByteDance/Hyper-SD"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class ModelVariant(StrEnum):
    SDXL_2STEP_LORA = "SDXL_2step_LoRA"
    SDXL_4STEP_LORA = "SDXL_4step_LoRA"
    SDXL_8STEP_LORA = "SDXL_8step_LoRA"


_LORA_FILES = {
    ModelVariant.SDXL_2STEP_LORA: "Hyper-SDXL-2steps-lora.safetensors",
    ModelVariant.SDXL_4STEP_LORA: "Hyper-SDXL-4steps-lora.safetensors",
    ModelVariant.SDXL_8STEP_LORA: "Hyper-SDXL-8steps-lora.safetensors",
}

_NUM_STEPS = {
    ModelVariant.SDXL_2STEP_LORA: 2,
    ModelVariant.SDXL_4STEP_LORA: 4,
    ModelVariant.SDXL_8STEP_LORA: 8,
}


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.SDXL_2STEP_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SDXL_4STEP_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SDXL_8STEP_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SDXL_2STEP_LORA

    prompt = "a photo of an astronaut riding a horse on mars"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Hyper-SD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            hf_hub_download(LORA_REPO, lora_file),
        )
        self.pipeline.fuse_lora()

        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing="trailing"
        )

        self.pipeline.to("cpu", dtype=torch.float32)
        for module in [
            self.pipeline.text_encoder,
            self.pipeline.unet,
            self.pipeline.text_encoder_2,
            self.pipeline.vae,
        ]:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(
            self.pipeline,
            self.prompt,
            num_inference_steps=_NUM_STEPS[self._variant],
        )

        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
