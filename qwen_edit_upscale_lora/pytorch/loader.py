# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Edit 2509 Upscale LoRA model loader implementation.

Loads the Qwen/Qwen-Image-Edit-2509 base pipeline and applies upscale
enhancement LoRA weights from vafipas663/Qwen-Edit-2509-Upscale-LoRA for
image upscaling and enhancement (up to 16x).

Two LoRA checkpoints are chained sequentially:
- LoRA A: qwen-edit-enhance_64-v3_000001000 (rank-64 base enhancement)
- LoRA B: qwen-edit-enhance_000004250 (refinement)

Available variants:
- UPSCALE_LORA: Full two-stage LoRA enhancement pipeline
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPlusPipeline  # type: ignore[import]

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "vafipas663/Qwen-Edit-2509-Upscale-LoRA"

# LoRA weight filenames (applied sequentially)
LORA_A = "qwen-edit-enhance_64-v3_000001000.safetensors"
LORA_B = "qwen-edit-enhance_000004250.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Edit 2509 Upscale LoRA variants."""

    UPSCALE_LORA = "Upscale_LoRA"


class ModelLoader(ForgeModel):
    """Qwen-Edit 2509 Upscale LoRA model loader."""

    _VARIANTS = {
        ModelVariant.UPSCALE_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.UPSCALE_LORA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_EDIT_UPSCALE_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image-Edit pipeline with upscale LoRA weights applied.

        Returns:
            torch.nn.Module: The transformer model from the pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        # Chain LoRA A then LoRA B sequentially
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_A,
            adapter_name="enhance_base",
        )
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_B,
            adapter_name="enhance_refine",
        )
        self.pipeline.set_adapters(["enhance_base", "enhance_refine"])

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        self._transformer = self.pipeline.transformer
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns:
            dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = next(self._transformer.parameters()).dtype
        batch_size = kwargs.get("batch_size", 1)

        img_dim = 64
        text_dim = 3584
        txt_seq_len = 32

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
