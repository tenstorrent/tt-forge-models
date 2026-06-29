# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step 1.5 audio VAE decoder loader (AutoencoderOobleck).

Output component: decodes the DiT's 64-channel acoustic latents (25 Hz) back to
48 kHz stereo audio. ``downsampling_ratios=[2,4,4,6,10]`` => 1920x upsample, so
a 250-frame latent (10 s @ 25 Hz) decodes to 480000 stereo samples (10 s).
"""
import torch
import torch.nn as nn
from typing import Optional

from diffusers import AutoencoderOobleck

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"


class _OobleckDecoderWrapper(nn.Module):
    """Exposes the latent->audio decode path as a single forward pass."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents).sample


class ModelVariant(StrEnum):
    """Available ACE-Step 1.5 VAE variants."""

    OOBLECK = "oobleck"


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 audio VAE decoder (AutoencoderOobleck) loader."""

    _VARIANTS = {
        ModelVariant.OOBLECK: ModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OOBLECK
    _SUBFOLDER = "vae"
    _LATENT_CHANNELS = 64
    _LATENT_FRAMES = 250  # 10 s @ 25 Hz

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="acestep_vae",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, **kwargs):
        vae = AutoencoderOobleck.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder=self._SUBFOLDER,
            revision=_REVISION,
        ).eval()
        wrapper = _OobleckDecoderWrapper(vae)
        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        g = torch.Generator().manual_seed(0)
        latents = torch.randn(
            batch_size, self._LATENT_CHANNELS, self._LATENT_FRAMES, generator=g
        ).to(dtype)
        return {"latents": latents}
