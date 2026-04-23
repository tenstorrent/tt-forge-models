# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fibo-Edit model loader implementation for image-to-image editing.

Fibo-Edit is an 8B-parameter DiT-based, flow-matching image editing model
from Bria AI. It consumes a source image plus a structured JSON prompt and
produces an edited image via the BriaFiboEditPipeline from diffusers.

briaai/Fibo-Edit is a gated HuggingFace repo. This loader constructs the
BriaFiboTransformer2DModel with random weights and provides synthetic inputs,
following the same pattern as flux_fp8. Dimensions are chosen so that:
  - text_encoder_dim=2048 matches the SmolLM3 hidden_size in the real model
  - joint_attention_dim=4096 = 2 * text_encoder_dim (pipeline cats last 2 layers)
  - in_channels=64 matches the Wan VAE latent packing (z_dim=16, pack=4)
"""

from typing import Any, Optional

import torch
from diffusers import BriaFiboTransformer2DModel

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

_NUM_LAYERS = 2
_NUM_SINGLE_LAYERS = 4
_NUM_ATTENTION_HEADS = 4
_ATTENTION_HEAD_DIM = 128
_IN_CHANNELS = 64
_JOINT_ATTENTION_DIM = 4096
_TEXT_ENCODER_DIM = 2048

_IMAGE_HEIGHT = 128
_IMAGE_WIDTH = 128
_VAE_SCALE_FACTOR = 8
_TEXT_SEQ_LEN = 16


class ModelVariant(StrEnum):
    """Available Fibo-Edit model variants."""

    FIBO_EDIT = "fibo_edit"


class ModelLoader(ForgeModel):
    """Fibo-Edit model loader implementation for image-to-image editing tasks."""

    _VARIANTS = {
        ModelVariant.FIBO_EDIT: ModelConfig(
            pretrained_model_name="briaai/Fibo-Edit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FIBO_EDIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="fibo_edit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BriaFiboTransformer2DModel with random weights.

        briaai/Fibo-Edit is a gated repo so the transformer is instantiated
        directly with a reduced-layer config rather than downloaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return BriaFiboTransformer2DModel(
            patch_size=1,
            in_channels=_IN_CHANNELS,
            num_layers=_NUM_LAYERS,
            num_single_layers=_NUM_SINGLE_LAYERS,
            attention_head_dim=_ATTENTION_HEAD_DIM,
            num_attention_heads=_NUM_ATTENTION_HEADS,
            joint_attention_dim=_JOINT_ATTENTION_DIM,
            pooled_projection_dim=None,
            guidance_embeds=False,
            axes_dims_rope=[16, 56, 56],
            text_encoder_dim=_TEXT_ENCODER_DIM,
        ).to(dtype)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate synthetic inputs for the BriaFiboTransformer2DModel.

        Shapes follow the pipeline's internal packing conventions:
          - Latent: VAE z_dim=16, packed 2x2 → in_channels=64
          - Text: joint_attention_dim=4096 = 2 * text_encoder_dim
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        num_channels_latents = _IN_CHANNELS // 4
        num_blocks = _NUM_LAYERS + _NUM_SINGLE_LAYERS

        height_latent = 2 * (_IMAGE_HEIGHT // (_VAE_SCALE_FACTOR * 2))
        width_latent = 2 * (_IMAGE_WIDTH // (_VAE_SCALE_FACTOR * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2
        seq_len = h_packed * w_packed

        hidden_states = torch.randn(batch_size, seq_len, _IN_CHANNELS, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, _TEXT_SEQ_LEN, _JOINT_ATTENTION_DIM, dtype=dtype
        )
        text_encoder_layers = [
            torch.randn(batch_size, _TEXT_SEQ_LEN, _TEXT_ENCODER_DIM, dtype=dtype)
            for _ in range(num_blocks)
        ]

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = torch.arange(h_packed)[:, None].float()
        latent_image_ids[..., 2] = torch.arange(w_packed)[None, :].float()
        img_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        txt_ids = torch.zeros(_TEXT_SEQ_LEN, 3, dtype=dtype)
        timestep = torch.tensor([1.0] * batch_size, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "text_encoder_layers": text_encoder_layers,
            "pooled_projections": None,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": None,
            "joint_attention_kwargs": {},
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
