# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
React Native ExecuTorch BK-SDM-Tiny text-to-image model loader implementation.

Note: The original model (software-mansion/react-native-executorch-bk-sdm-tiny) is in
ExecuTorch .pte format intended for mobile inference. Since ExecuTorch format is not
compatible with PyTorch, this loader uses the upstream base model
(vivym/bk-sdm-tiny-vpred) via StableDiffusionPipeline.
"""

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing


class ModelVariant(StrEnum):
    """Available React Native ExecuTorch BK-SDM-Tiny model variants."""

    BK_SDM_TINY = "bk-sdm-tiny"


class ModelLoader(ForgeModel):
    """React Native ExecuTorch BK-SDM-Tiny text-to-image model loader implementation."""

    _VARIANTS = {
        ModelVariant.BK_SDM_TINY: ModelConfig(
            pretrained_model_name="vivym/bk-sdm-tiny-vpred",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BK_SDM_TINY

    prompt = "a tropical bird sitting on a branch"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="React Native ExecuTorch BK-SDM-Tiny",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the BK-SDM-Tiny UNet from the pipeline.

        Returns:
            torch.nn.Module: The UNet used for denoising.
        """
        self.pipeline = load_pipe(self._variant_config.pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare UNet tensor inputs for a single forward pass.

        Returns:
            tuple: (latents, timestep, encoder_hidden_states)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        latents, timestep, encoder_hidden_states = stable_diffusion_preprocessing(
            self.pipeline, self.prompt
        )

        if dtype_override is not None:
            latents = latents.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return latents, timestep, encoder_hidden_states
