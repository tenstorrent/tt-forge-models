# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shap-E model loader implementation for text-to-3D generation.
"""

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from diffusers import ShapEPipeline


class ModelVariant(StrEnum):
    """Available Shap-E model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Shap-E model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="openai/shap-e",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Shap-E",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Shap-E prior transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The pre-trained Shap-E prior transformer.
        """
        dtype = dtype_override or torch.float32
        pipe = ShapEPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline = pipe
        return pipe.prior

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Shap-E prior transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Number of samples in the batch.

        Returns:
            list: [hidden_states, timestep, proj_embedding] tensors for the prior.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        pipe = self.pipeline
        dtype = dtype_override or torch.float32
        device = "cpu"

        prompt_embeds = pipe._encode_prompt(
            "a shark",
            device=device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=False,
        )

        num_embeddings = pipe.prior.config.num_embeddings
        embedding_dim = pipe.prior.config.embedding_dim
        latents = torch.randn(batch_size, num_embeddings, embedding_dim, dtype=dtype)

        pipe.scheduler.set_timesteps(1, device=device)
        t = pipe.scheduler.timesteps[0]
        scaled_model_input = pipe.scheduler.scale_model_input(latents, t)

        return [scaled_model_input, t, prompt_embeds]
