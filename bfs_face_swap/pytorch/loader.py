# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BFS (Best Face Swap) LoRA model loader implementation.

Loads the Qwen-Image-Edit diffusion transformer with BFS LoRA weights
from Alissonerdx/BFS-Best-Face-Swap applied and merged. Returns the
transformer component for testing.

Available variants:
- HEAD_V5_2511: Head swap v5 on Qwen-Image-Edit 2511 (recommended)
- HEAD_V3_2509: Head swap v3 on Qwen-Image-Edit 2509
- FACE_V1_2509: Face-only swap v1 on Qwen-Image-Edit 2509
"""

from typing import Any, Dict, Optional

import torch
from diffusers import QwenImageEditPipeline

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

LORA_REPO = "Alissonerdx/BFS-Best-Face-Swap"

# Base model repos per Qwen-Image-Edit version
_BASE_MODELS = {
    "2509": "Qwen/Qwen-Image-Edit-2509",
    "2511": "Qwen/Qwen-Image-Edit-2511",
}


class ModelVariant(StrEnum):
    """Available BFS face swap LoRA variants."""

    HEAD_V5_2511 = "Head_V5_2511"
    HEAD_V3_2509 = "Head_V3_2509"
    FACE_V1_2509 = "Face_V1_2509"


# LoRA weight filenames per variant
_LORA_FILES = {
    ModelVariant.HEAD_V5_2511: "bfs_head_v5_2511_original.safetensors",
    ModelVariant.HEAD_V3_2509: "bfs_head_v3_qwen_image_edit_2509.safetensors",
    ModelVariant.FACE_V1_2509: "bfs_face_v1_qwen_image_edit_2509.safetensors",
}


class ModelLoader(ForgeModel):
    """BFS (Best Face Swap) LoRA model loader."""

    _VARIANTS = {
        ModelVariant.HEAD_V5_2511: ModelConfig(
            pretrained_model_name=_BASE_MODELS["2511"],
        ),
        ModelVariant.HEAD_V3_2509: ModelConfig(
            pretrained_model_name=_BASE_MODELS["2509"],
        ),
        ModelVariant.FACE_V1_2509: ModelConfig(
            pretrained_model_name=_BASE_MODELS["2509"],
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HEAD_V5_2511

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BFS_FACE_SWAP",
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
        """Load the Qwen-Image-Edit transformer with BFS LoRA weights merged.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self._transformer is None:
            pipe = QwenImageEditPipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            pipe.load_lora_weights(
                LORA_REPO,
                weight_name=_LORA_FILES[self._variant],
            )
            pipe.fuse_lora()
            self._transformer = pipe.transformer
            self._transformer.eval()
            del pipe
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)

        return self._transformer

    def load_inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare sample tensor inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = torch.float32

        # Transformer config: in_channels=64, joint_attention_dim=3584
        img_dim = 64
        text_dim = 3584
        txt_seq_len = 32

        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(1, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(1, txt_seq_len, text_dim, dtype=dtype)
        encoder_hidden_states_mask = torch.ones(1, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0], dtype=dtype)
        img_shapes = [(frame, height, width)]

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
