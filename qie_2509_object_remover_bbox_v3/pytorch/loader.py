# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QIE-2509-Object-Remover-Bbox-v3 LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2509 base diffusion pipeline and applies
the prithivMLmods/QIE-2509-Object-Remover-Bbox-v3 LoRA adapter for
bounding-box guided object removal from images.

Available variants:
- QIE_2509_OBJECT_REMOVER_BBOX_V3: Object Remover (Bbox) LoRA
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPipeline

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

BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO_ID = "prithivMLmods/QIE-2509-Object-Remover-Bbox-v3"
LORA_WEIGHT_NAME = "QIE-2509-Object-Remover-Bbox-v3-10000.safetensors"


class ModelVariant(StrEnum):
    """Available QIE-2509-Object-Remover-Bbox-v3 model variants."""

    QIE_2509_OBJECT_REMOVER_BBOX_V3 = "Object_Remover_Bbox_v3"


class ModelLoader(ForgeModel):
    """QIE-2509-Object-Remover-Bbox-v3 LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QIE_2509_OBJECT_REMOVER_BBOX_V3: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QIE_2509_OBJECT_REMOVER_BBOX_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QIE_2509_Object_Remover_Bbox_v3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen-Image-Edit-2509 transformer with Object Remover Bbox v3 LoRA.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        pipeline = QwenImageEditPipeline.from_pretrained(
            BASE_REPO_ID,
            torch_dtype=dtype,
        )
        pipeline.load_lora_weights(
            LORA_REPO_ID,
            weight_name=LORA_WEIGHT_NAME,
        )
        pipeline.fuse_lora()

        self._transformer = pipeline.transformer.to(dtype=dtype)
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
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
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
