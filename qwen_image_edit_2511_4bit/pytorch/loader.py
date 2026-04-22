# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ovedrive/Qwen-Image-Edit-2511-4bit model loader implementation.

Loads the NF4-quantized diffusion transformer from
ovedrive/Qwen-Image-Edit-2511-4bit, a bitsandbytes 4-bit quantized version of
Qwen/Qwen-Image-Edit-2511 for image editing.

Available variants:
- QWEN_IMAGE_EDIT_2511_4BIT: NF4-quantized Qwen-Image-Edit-2511 transformer (bf16 compute)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageTransformer2DModel

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

REPO_ID = "ovedrive/Qwen-Image-Edit-2511-4bit"


class ModelVariant(StrEnum):
    """Available ovedrive/Qwen-Image-Edit-2511-4bit model variants."""

    QWEN_IMAGE_EDIT_2511_4BIT = "Edit_2511_4bit"


def _dequantize_bnb_model(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Replace bitsandbytes Linear4bit layers with dequantized nn.Linear layers in-place."""
    import bitsandbytes.functional as F
    import bitsandbytes.nn as bnb

    for name, module in list(model.named_modules()):
        if not isinstance(module, bnb.Linear4bit):
            continue
        weight = F.dequantize_nf4(module.weight, module.weight.quant_state).to(dtype)
        new_linear = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            dtype=dtype,
        )
        new_linear.weight = torch.nn.Parameter(weight)
        if module.bias is not None:
            new_linear.bias = torch.nn.Parameter(module.bias.to(dtype))
        # Traverse to the parent module and replace the child
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_linear)


class ModelLoader(ForgeModel):
    """ovedrive/Qwen-Image-Edit-2511-4bit model loader for the NF4-quantized diffusion transformer."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_2511_4BIT: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_2511_4BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_2511_4BIT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> QwenImageTransformer2DModel:
        """Load the NF4-quantized diffusion transformer and dequantize for TT execution."""
        from diffusers.quantizers.bitsandbytes import bnb_quantizer

        # Bypass the CUDA GPU check: this model ships with BnB NF4 weights which
        # we dequantize immediately after loading for Tenstorrent execution.
        original_validate = bnb_quantizer.BnB4BitDiffusersQuantizer.validate_environment
        bnb_quantizer.BnB4BitDiffusersQuantizer.validate_environment = (
            lambda *a, **kw: None
        )
        try:
            transformer = QwenImageTransformer2DModel.from_pretrained(
                REPO_ID,
                subfolder="transformer",
                torch_dtype=dtype,
                device_map="cpu",
            )
        finally:
            bnb_quantizer.BnB4BitDiffusersQuantizer.validate_environment = (
                original_validate
            )

        _dequantize_bnb_model(transformer, dtype)
        transformer.eval()
        self._transformer = transformer
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the NF4-quantized Qwen-Image-Edit-2511 diffusion transformer.

        Returns:
            QwenImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64 (img_in linear input dimension)
        img_dim = 64
        # joint_attention_dim from config = 3584
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
        # img_shapes: list of lists of (frame, height, width) tuples per batch item.
        # Each batch item is a list of layer shapes; zero_cond_t mode requires this nested format.
        img_shapes = [[(frame, height, width)] for _ in range(batch_size)]

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
