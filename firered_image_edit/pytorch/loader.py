# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FireRed-Image-Edit 1.1 pipeline model loader implementation.

Loads the full FireRed-Image-Edit-1.1 diffusion pipeline for instruction-guided
image editing. The model is built on the Qwen-Image-Edit-Plus architecture and
accepts a source image plus a text prompt describing the desired edit.

Available variants:
- FIRERED_IMAGE_EDIT_1_1: FireRed-Image-Edit 1.1 diffusers pipeline
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPlusPipeline

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

REPO_ID = "FireRedTeam/FireRed-Image-Edit-1.1"


class ModelVariant(StrEnum):
    """Available FireRed-Image-Edit pipeline model variants."""

    FIRERED_IMAGE_EDIT_1_1 = "Edit_1.1"


class ModelLoader(ForgeModel):
    """FireRed-Image-Edit 1.1 pipeline model loader."""

    _VARIANTS = {
        ModelVariant.FIRERED_IMAGE_EDIT_1_1: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FIRERED_IMAGE_EDIT_1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIRERED_IMAGE_EDIT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the FireRed-Image-Edit 1.1 pipeline and return its transformer.

        Returns:
            QwenImageTransformer2DModel: the diffusion transformer nn.Module.
        """
        dtype = dtype_override or torch.bfloat16
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare synthetic tensor inputs for the QwenImageTransformer2DModel.

        Constructs inputs matching the transformer's forward() signature using a
        512x512 reference image size, one conditioning image, and a short text
        sequence of 128 tokens.
        """
        if self.pipeline is None:
            self.load_model()

        transformer = self.pipeline.transformer
        dtype = transformer.dtype

        # Spatial dimensions for 512x512 images.
        height, width = 512, 512
        vae_scale_factor = self.pipeline.vae_scale_factor  # 8 for this VAE
        latent_h = 2 * (height // (vae_scale_factor * 2))  # 64
        latent_w = 2 * (width // (vae_scale_factor * 2))  # 64

        # Packed sequence length per image: (latent_h//2) * (latent_w//2)
        seq_len = (latent_h // 2) * (latent_w // 2)  # 1024
        in_channels = transformer.config.in_channels  # 64

        # Concatenated noisy latents + conditioning image latents on seq dim.
        hidden_states = torch.randn(1, seq_len * 2, in_channels, dtype=dtype)

        # Text encoder outputs from Qwen2.5-VL (hidden_size = joint_attention_dim).
        text_seq_len = 128
        joint_dim = transformer.config.joint_attention_dim
        encoder_hidden_states = torch.randn(1, text_seq_len, joint_dim, dtype=dtype)
        encoder_hidden_states_mask = torch.ones(1, text_seq_len, dtype=torch.long)

        # Normalised timestep (pipeline divides by 1000 before passing).
        timestep = torch.tensor([0.5], dtype=dtype)

        # Image shape metadata: [(num_imgs, packed_h, packed_w)] per batch item.
        packed_h = latent_h // 2  # 32
        packed_w = latent_w // 2  # 32
        img_shapes = [[(1, packed_h, packed_w), (1, packed_h, packed_w)]]

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
