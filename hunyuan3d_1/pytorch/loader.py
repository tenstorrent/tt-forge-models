# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D-1 model loader implementation for image-to-3D generation.

Hunyuan3D-1.0 is Tencent's unified framework for text-to-3D and image-to-3D
generation. The first stage is a Zero123Plus-style multi-view diffusion model
that produces consistent multi-view RGB renderings. This loader exposes the
UNet2DConditionModel backbone from that multi-view diffusion stage, available
in either the lite (SD2-derived) or std (SDXL-derived) flavor.
"""

from typing import Optional

import torch
from diffusers import UNet2DConditionModel

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

REPO_ID = "tencent/Hunyuan3D-1"


class ModelVariant(StrEnum):
    """Available Hunyuan3D-1 model variants."""

    MVD_LITE = "mvd_lite"
    MVD_STD = "mvd_std"


# Architecture parameters per variant UNet config.
# mvd_lite: cross_attention_dim=1024 (SD 2 derivative)
# mvd_std:  cross_attention_dim=2048 (SDXL derivative, text_time added cond)
_VARIANT_PARAMS = {
    ModelVariant.MVD_LITE: {
        "subfolder": "mvd_lite/unet",
        "cross_attention_dim": 1024,
        "sample_size": 96,
        "added_cond_kwargs": None,
    },
    ModelVariant.MVD_STD: {
        "subfolder": "mvd_std/unet",
        "cross_attention_dim": 2048,
        "sample_size": 96,
        # SDXL-style text_time conditioning: text_embeds [B, 1280], time_ids [B, 6]
        "added_cond_kwargs": {"text_embeds_dim": 1280, "time_ids_dim": 6},
    },
}

# Latent channels for the UNet (in_channels/out_channels == 4).
LATENT_CHANNELS = 4
# Text encoder sequence length used for sample conditioning.
TEXT_SEQ_LEN = 77


class ModelLoader(ForgeModel):
    """Hunyuan3D-1 multi-view diffusion UNet loader."""

    _VARIANTS = {
        ModelVariant.MVD_LITE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.MVD_STD: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MVD_LITE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Hunyuan3D-1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hunyuan3D-1 multi-view UNet.

        Returns:
            UNet2DConditionModel: The pre-trained multi-view diffusion UNet.
        """
        variant = self._variant or self.DEFAULT_VARIANT
        params = _VARIANT_PARAMS[variant]

        model_kwargs = dict(kwargs)
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = UNet2DConditionModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder=params["subfolder"],
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the Hunyuan3D-1 multi-view UNet.

        Returns:
            dict: (sample, timestep, encoder_hidden_states[, added_cond_kwargs]).
        """
        dtype = dtype_override or torch.float32
        variant = self._variant or self.DEFAULT_VARIANT
        params = _VARIANT_PARAMS[variant]

        sample_size = params["sample_size"]
        sample = torch.randn(
            batch_size,
            LATENT_CHANNELS,
            sample_size,
            sample_size,
            dtype=dtype,
        )
        timestep = torch.tensor([1], dtype=torch.long)
        encoder_hidden_states = torch.randn(
            batch_size,
            TEXT_SEQ_LEN,
            params["cross_attention_dim"],
            dtype=dtype,
        )

        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        added = params["added_cond_kwargs"]
        if added is not None:
            inputs["added_cond_kwargs"] = {
                "text_embeds": torch.randn(
                    batch_size, added["text_embeds_dim"], dtype=dtype
                ),
                "time_ids": torch.zeros(batch_size, added["time_ids_dim"], dtype=dtype),
            }

        return inputs
