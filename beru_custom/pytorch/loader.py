# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
beru_custom (jagat334433/beru_custom) model loader implementation.

beru_custom is a custom Stable Diffusion XL checkpoint distributed as
single-file safetensors in the jagat334433/beru_custom HuggingFace repo.

Available variants:
- BERU_CUSTOM_2: beru_custom_2.safetensors SDXL checkpoint
- BERU_CUSTOM_MERGE: beru_custom_merge.safetensors SDXL merged checkpoint
"""

from typing import Optional

import torch
from huggingface_hub import hf_hub_download

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl

REPO_ID = "jagat334433/beru_custom"


class ModelVariant(StrEnum):
    """Available beru_custom model variants."""

    BERU_CUSTOM_2 = "beru_custom_2"
    BERU_CUSTOM_MERGE = "beru_custom_merge"


_VARIANT_CHECKPOINTS = {
    ModelVariant.BERU_CUSTOM_2: "beru_custom_2.safetensors",
    ModelVariant.BERU_CUSTOM_MERGE: "beru_custom_merge.safetensors",
}


class ModelLoader(ForgeModel):
    """beru_custom SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.BERU_CUSTOM_2: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.BERU_CUSTOM_MERGE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BERU_CUSTOM_2

    prompt = (
        "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="beru_custom",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the beru_custom SDXL pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        checkpoint_file = _VARIANT_CHECKPOINTS[self._variant]
        model_path = hf_hub_download(repo_id=REPO_ID, filename=checkpoint_file)
        self.pipeline = load_pipe(model_path)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the beru_custom SDXL UNet model.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

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
