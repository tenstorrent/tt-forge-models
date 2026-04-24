# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QIE-2511-Outfit-Design-Layout LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2511 base diffusion pipeline and applies
the prithivMLmods/QIE-2511-Outfit-Design-Layout LoRA adapter for adding
outfit elements or design layouts into marked image regions.

Available variants:
- QIE_2511_OUTFIT_DESIGN_LAYOUT: Outfit Design Layout LoRA (bf16)
"""

from typing import Any, Dict, Optional

import torch
from diffusers import DiffusionPipeline

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

BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO_ID = "prithivMLmods/QIE-2511-Outfit-Design-Layout"


class ModelVariant(StrEnum):
    """Available QIE-2511-Outfit-Design-Layout model variants."""

    QIE_2511_OUTFIT_DESIGN_LAYOUT = "Outfit_Design_Layout"


class ModelLoader(ForgeModel):
    """QIE-2511-Outfit-Design-Layout LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QIE_2511_OUTFIT_DESIGN_LAYOUT: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QIE_2511_OUTFIT_DESIGN_LAYOUT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QIE_2511_Outfit_Design_Layout",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen-Image-Edit-2511 transformer with Outfit Design Layout LoRA.

        Returns:
            QwenImageTransformer2DModel: The transformer with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        pipe = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        pipe.load_lora_weights(LORA_REPO_ID)
        pipe.fuse_lora()
        self._transformer = pipe.transformer
        self._transformer.eval()
        del pipe
        return self._transformer

    def load_inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare synthetic tensor inputs for the QwenImageTransformer2DModel.

        Uses a small 64x64 image size. The pipeline concatenates noisy latents
        with reference image latents along the sequence dimension, so
        hidden_states has twice the per-image sequence length.

        Returns:
            dict matching QwenImageTransformer2DModel.forward() signature.
        """
        if self._transformer is None:
            self.load_model()

        dtype = next(self._transformer.parameters()).dtype
        batch_size = 1

        # For a 64x64 image: vae_scale_factor=8, patch_size=2
        # packed spatial dims: 64 // (8 * 2) = 4
        frame, h_packed, w_packed = 1, 4, 4
        in_channels = self._transformer.config.in_channels  # 64
        joint_attention_dim = self._transformer.config.joint_attention_dim  # 3584

        img_seq_per = frame * h_packed * w_packed  # 16 tokens per image
        # editing pipeline concatenates noisy latents + reference image latents
        total_img_seq = img_seq_per * 2

        hidden_states = torch.randn(batch_size, total_img_seq, in_channels, dtype=dtype)

        txt_seq_len = 32
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, joint_attention_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([0.5] * batch_size, dtype=dtype)
        img_shapes = [
            [(frame, h_packed, w_packed), (frame, h_packed, w_packed)]
        ] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
