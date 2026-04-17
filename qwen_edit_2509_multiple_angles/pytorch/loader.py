# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Edit-2509-Multiple-angles LoRA model loader implementation.

Loads the Qwen-Image-Edit-2509 base pipeline and applies the
dx8152/Qwen-Edit-2509-Multiple-angles LoRA weights for camera movement
and angle control in image editing.

Available variants:
- QWEN_EDIT_2509_MULTIPLE_ANGLES: Camera angle control LoRA
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
LORA_REPO = "dx8152/Qwen-Edit-2509-Multiple-angles"


class ModelVariant(StrEnum):
    """Available Qwen-Edit-2509-Multiple-angles variants."""

    QWEN_EDIT_2509_MULTIPLE_ANGLES = "Edit_2509_MultipleAngles"


class ModelLoader(ForgeModel):
    """Qwen-Edit-2509-Multiple-angles LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_EDIT_2509_MULTIPLE_ANGLES: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_EDIT_2509_MULTIPLE_ANGLES

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_EDIT_2509_MULTIPLE_ANGLES",
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
        """Load the Qwen-Image-Edit-2509 pipeline with Multiple-angles LoRA.

        Returns:
            torch.nn.Module: The transformer model from the pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns:
            dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

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
