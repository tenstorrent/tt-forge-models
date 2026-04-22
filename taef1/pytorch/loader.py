# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TAEF1 tiny autoencoder model loader implementation
"""

import torch
from diffusers import AutoencoderTiny
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


class TAEF1Wrapper(torch.nn.Module):
    """Wraps AutoencoderTiny to fix dtype mismatch from internal uint8 quantization.

    AutoencoderTiny.forward converts latents to uint8 and back; the division
    `scaled_enc / 255.0` returns float32 even when the model is bfloat16,
    causing a Conv2d input/bias dtype mismatch in the decoder.
    """

    def __init__(self, model: AutoencoderTiny):
        super().__init__()
        self.model = model

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        dtype = sample.dtype
        enc = self.model.encode(sample).latents
        scaled_enc = self.model.scale_latents(enc).mul_(255).round_().byte()
        unscaled_enc = self.model.unscale_latents(scaled_enc / 255.0).to(dtype)
        return self.model.decode(unscaled_enc).sample


class ModelVariant(StrEnum):
    """Available TAEF1 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """TAEF1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="madebyollin/taef1",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TAEF1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TAEF1 tiny autoencoder model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            TAEF1Wrapper: The pre-trained TAEF1 VAE model wrapped to fix dtype issues.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = AutoencoderTiny.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype
        )
        model.eval()
        return TAEF1Wrapper(model).eval()

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the TAEF1 model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            torch.Tensor: Random image tensor suitable for VAE encoding.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        # TAEF1 expects 3-channel images; use 256x256 as a reasonable default.
        return torch.randn(batch_size, 3, 256, 256, dtype=dtype)
