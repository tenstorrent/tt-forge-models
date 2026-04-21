# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TripoSG model loader implementation for image-to-3D generation.

Loads the TripoSGDiTModel (DiT backbone) from the TripoSG pipeline, which is
the rectified-flow transformer for generating 3D shapes from image conditioning.

Requires the TripoSG repository to be cloned at /tmp/triposg_repo.
"""
import os
import sys
from typing import Optional

import torch

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

TRIPOSG_REPO_PATH = "/tmp/triposg_repo"


def _ensure_triposg_importable():
    """Ensure the TripoSG repo is cloned and importable."""
    if "triposg" not in sys.modules:
        if not os.path.isdir(TRIPOSG_REPO_PATH):
            import subprocess

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/VAST-AI-Research/TripoSG.git",
                    TRIPOSG_REPO_PATH,
                ]
            )

        if TRIPOSG_REPO_PATH not in sys.path:
            sys.path.insert(0, TRIPOSG_REPO_PATH)


class ModelVariant(StrEnum):
    """Available TripoSG model variants."""

    V1 = "V1"


class ModelLoader(ForgeModel):
    """TripoSG model loader for the TripoSGDiTModel (DiT backbone)."""

    _VARIANTS = {
        ModelVariant.V1: ModelConfig(
            pretrained_model_name="VAST-AI/TripoSG",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1

    # Architecture parameters from transformer/config.json
    _IN_CHANNELS = 64
    _CROSS_ATTENTION_DIM = 1024
    # TripoSG operates on 2048 latent tokens
    _NUM_LATENTS = 2048
    # DINOv2 ViT-L/14 @ 518px: (518/14)^2 + 1 = 1370
    _COND_SEQ_LEN = 1370

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TripoSG",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TripoSG DiT model.

        Returns:
            torch.nn.Module: The TripoSGDiTModel rectified-flow transformer.
        """
        _ensure_triposg_importable()
        from triposg.models.transformers.triposg_transformer import TripoSGDiTModel

        model = TripoSGDiTModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the TripoSGDiTModel.

        Returns:
            dict: Input tensors for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # hidden_states: noisy latent tokens [B, num_latents, in_channels]
        hidden_states = torch.randn(
            batch_size,
            self._NUM_LATENTS,
            self._IN_CHANNELS,
            dtype=dtype,
        )

        # timestep: diffusion timestep [B]
        timestep = torch.full((batch_size,), 0.5, dtype=dtype)

        # encoder_hidden_states: DINOv2 image conditioning tokens
        encoder_hidden_states = torch.randn(
            batch_size,
            self._COND_SEQ_LEN,
            self._CROSS_ATTENTION_DIM,
            dtype=dtype,
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
