# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QIE-2509-Object-Remover-Bbox LoRA image-to-image model loader implementation.

Loads the Qwen-Image-Edit-2509 base diffusion pipeline, applies the
prithivMLmods/QIE-2509-Object-Remover-Bbox LoRA weights, fuses them, and
returns the transformer component for compilation.
"""

import torch
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


class ModelVariant(StrEnum):
    """Available QIE-2509-Object-Remover-Bbox model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """QIE-2509-Object-Remover-Bbox LoRA image-to-image model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="prithivMLmods/QIE-2509-Object-Remover-Bbox",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    base_model = "Qwen/Qwen-Image-Edit-2509"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QIE-2509-Object-Remover-Bbox",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the QIE-2509-Object-Remover-Bbox transformer with fused LoRA weights.

        Loads the base Qwen-Image-Edit-2509 pipeline, applies and fuses the
        Object-Remover-Bbox LoRA adapter weights, then returns the transformer
        component as a torch.nn.Module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        from diffusers import DiffusionPipeline

        dtype = dtype_override or torch.bfloat16

        if self._transformer is None:
            pipe = DiffusionPipeline.from_pretrained(
                self.base_model, torch_dtype=dtype, **kwargs
            )
            pipe.load_lora_weights(self._variant_config.pretrained_model_name)
            pipe.fuse_lora()
            self._transformer = pipe.transformer
            self._transformer.eval()
            del pipe
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)

        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load sample inputs for the QIE-2509-Object-Remover-Bbox transformer.

        Returns inputs matching QwenImageTransformer2DModel.forward() signature.

        Args:
            dtype_override: Optional torch.dtype for input tensors.
            batch_size: Number of samples in the batch.

        Returns:
            dict: Keyword arguments for QwenImageTransformer2DModel.forward().
        """
        dtype = dtype_override or torch.bfloat16

        # Transformer config: in_channels=64, joint_attention_dim=3584
        img_dim = 64
        text_dim = 3584
        txt_seq_len = 32

        # img_seq_len must equal frame * height * width for positional encoding
        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
