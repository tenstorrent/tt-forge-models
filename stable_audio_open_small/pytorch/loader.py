# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio Open Small text-to-audio model loader.

The small variant is distributed as a stable-audio-tools checkpoint rather than
a diffusers pipeline, so loading goes through
``stable_audio_tools.get_pretrained_model``. The DiT transformer inside the
returned wrapper is exposed as the model under test.
"""

from typing import Optional

import torch

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


class ModelVariant(StrEnum):
    """Available Stable Audio Open Small model variants."""

    SMALL = "small"


class ModelLoader(ForgeModel):
    """Stable Audio Open Small DiT transformer loader."""

    _VARIANTS = {
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="stabilityai/stable-audio-open-small",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SMALL

    # Sample prompt used when building text conditioning inputs.
    prompt = "128 BPM tech house drum loop"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._wrapper = None
        self._model_config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="StableAudioOpenSmall",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pretrained(self):
        """Download and instantiate the full stable-audio-tools wrapper."""
        from stable_audio_tools import get_pretrained_model

        self._wrapper, self._model_config = get_pretrained_model(
            self._variant_config.pretrained_model_name
        )
        self._wrapper.eval()
        return self._wrapper, self._model_config

    def load_model(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return the DiT transformer from the stable-audio-tools wrapper."""
        if self._wrapper is None:
            self._load_pretrained()

        transformer = self._wrapper.model.model
        if dtype_override is not None:
            transformer = transformer.to(dtype_override)
        return transformer

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a dict of sample inputs that match the DiT forward signature."""
        if self._wrapper is None:
            self._load_pretrained()

        dtype = dtype_override if dtype_override is not None else torch.float32

        sample_rate = self._model_config["sample_rate"]
        sample_size = self._model_config["sample_size"]
        seconds_total = float(sample_size) / float(sample_rate)

        # Build real conditioning tensors via the wrapper's conditioner so that
        # text embedding shapes are correct for the DiT forward pass.
        conditioning = [
            {
                "prompt": self.prompt,
                "seconds_start": 0.0,
                "seconds_total": seconds_total,
            }
        ]
        conditioner_outputs = self._wrapper.conditioner(conditioning, device="cpu")
        conditioning_inputs = self._wrapper.get_conditioning_inputs(conditioner_outputs)

        # get_conditioning_inputs uses the DiTWrapper's parameter names, while
        # the underlying DiffusionTransformer expects slightly different keys.
        key_renames = {
            "cross_attn_mask": "cross_attn_cond_mask",
            "global_cond": "global_embed",
        }

        # Prepare synthetic latents for the DiT using shapes read from the
        # model config rather than the transformer module, since the DiT does
        # not expose io_channels as an attribute.
        model_cfg = self._model_config["model"]
        io_channels = model_cfg["io_channels"]
        downsampling_ratio = model_cfg["pretransform"]["config"]["downsampling_ratio"]
        latent_length = sample_size // downsampling_ratio

        batch = 1
        x = torch.randn(batch, io_channels, latent_length, dtype=dtype)
        t = torch.rand(batch, dtype=dtype)

        inputs = {"x": x, "t": t}
        for key, value in conditioning_inputs.items():
            target_key = key_renames.get(key, key)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                inputs[target_key] = value.to(dtype=dtype)
            else:
                inputs[target_key] = value
        return inputs
