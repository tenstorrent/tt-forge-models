#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Self-Forcing (gdhe17/Self-Forcing) loader implementation.

Loads the Wan 2.1 T2V 1.3B base pipeline and applies the Self-Forcing
autoregressive checkpoint (generator_ema weights) to the transformer for
chunk-wise streaming text-to-video generation.

Available variants:
- SELF_FORCING_DMD: Distribution Matching Distillation checkpoint (primary)
- SELF_FORCING_GAN: GAN-trained checkpoint
- SELF_FORCING_SID: Score Identity Distillation checkpoint
- SELF_FORCING_SID_V2: SID v2 checkpoint
- SELF_FORCING_10S: Extended 10-second generation checkpoint
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel  # type: ignore[import]
from huggingface_hub import hf_hub_download  # type: ignore[import]

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

BASE_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SELF_FORCING_REPO = "gdhe17/Self-Forcing"

TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Self-Forcing checkpoint variants."""

    SELF_FORCING_DMD = "self_forcing_dmd"
    SELF_FORCING_GAN = "self_forcing_gan"
    SELF_FORCING_SID = "self_forcing_sid"
    SELF_FORCING_SID_V2 = "self_forcing_sid_v2"
    SELF_FORCING_10S = "self_forcing_10s"


_CHECKPOINT_FILES = {
    ModelVariant.SELF_FORCING_DMD: "checkpoints/self_forcing_dmd.pt",
    ModelVariant.SELF_FORCING_GAN: "checkpoints/self_forcing_gan.pt",
    ModelVariant.SELF_FORCING_SID: "checkpoints/self_forcing_sid.pt",
    ModelVariant.SELF_FORCING_SID_V2: "checkpoints/self_forcing_sid_v2.pt",
    ModelVariant.SELF_FORCING_10S: "checkpoints/self_forcing_10s.pt",
}


class ModelLoader(ForgeModel):
    """Self-Forcing loader that wraps Wan 2.1 T2V 1.3B with distilled weights."""

    _VARIANTS = {
        ModelVariant.SELF_FORCING_DMD: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SELF_FORCING_GAN: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SELF_FORCING_SID: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SELF_FORCING_SID_V2: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SELF_FORCING_10S: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SELF_FORCING_DMD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer: Optional[WanTransformer3DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SELF_FORCING",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ) -> WanTransformer3DModel:
        """Load the Wan 2.1 T2V 1.3B transformer with Self-Forcing weights.

        Downloads the selected `.pt` checkpoint from gdhe17/Self-Forcing and
        loads its ``generator_ema`` state dict into the pipeline's transformer.

        Returns:
            WanTransformer3DModel with Self-Forcing weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        vae = AutoencoderKLWan.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipeline = WanPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            vae=vae,
            torch_dtype=dtype,
        )

        ckpt_path = hf_hub_download(
            repo_id=SELF_FORCING_REPO,
            filename=_CHECKPOINT_FILES[self._variant],
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        generator_state = state_dict.get("generator_ema", state_dict)
        pipeline.transformer.load_state_dict(generator_state, strict=False)

        self._transformer = pipeline.transformer
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare tensor inputs for the WanTransformer3DModel forward pass."""
        if self._transformer is None:
            self.load_model()

        dtype = torch.bfloat16
        config = self._transformer.config

        return {
            "hidden_states": torch.randn(
                1,
                config.in_channels,
                TRANSFORMER_NUM_FRAMES,
                TRANSFORMER_HEIGHT,
                TRANSFORMER_WIDTH,
                dtype=dtype,
            ),
            "encoder_hidden_states": torch.randn(
                1,
                TRANSFORMER_TEXT_SEQ_LEN,
                config.text_dim,
                dtype=dtype,
            ),
            "timestep": torch.tensor([500], dtype=torch.long),
            "return_dict": False,
        }
