# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit LoRA Collection model loader implementation.

Loads the Qwen/Qwen-Image-Edit-2511 base pipeline and applies LoRA adapters
from strangerzonehf/Qwen-Image-Edit-LoRA-Collection for specialized image
editing tasks.

Available variants:
- OBJECT_REMOVER_BBOX: Remove objects via bounding box prompts
- GUIDED_HEAD_FACE_SWAP: Guided head/face swapping
- BW_TO_TRUE_COLOR: Convert black-and-white images to true color
- ANIME: Anime style transfer
- UNBLUR_ANYTHING: Unblur/sharpen images
"""

from typing import Any, Dict, Optional

import torch
from diffusers import QwenImageEditPlusPipeline

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "strangerzonehf/Qwen-Image-Edit-LoRA-Collection"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit LoRA Collection variants."""

    OBJECT_REMOVER_BBOX = "ObjectRemoverBbox"
    GUIDED_HEAD_FACE_SWAP = "GuidedHeadFaceSwap"
    BW_TO_TRUE_COLOR = "BW2TrueColor"
    ANIME = "Anime"
    UNBLUR_ANYTHING = "UnblurAnything"


_LORA_FILES = {
    ModelVariant.OBJECT_REMOVER_BBOX: "QIE-2511-Object-Remover-Bbox-5000.safetensors",
    ModelVariant.GUIDED_HEAD_FACE_SWAP: "QIE-2511-Guided-Head-Face-Swap-3000.safetensors",
    ModelVariant.BW_TO_TRUE_COLOR: "QIE-2511-BW2TrueColor-3000.safetensors",
    ModelVariant.ANIME: "Qwen-Image-Edit-2511-Anime-2000.safetensors",
    ModelVariant.UNBLUR_ANYTHING: "Qwen-Image-Edit-2511-Unblur-Anything.safetensors",
}

_PROMPTS = {
    ModelVariant.OBJECT_REMOVER_BBOX: "Remove the object in the bounding box",
    ModelVariant.GUIDED_HEAD_FACE_SWAP: "Swap the face with the reference",
    ModelVariant.BW_TO_TRUE_COLOR: "Convert this black and white image to true color",
    ModelVariant.ANIME: "Convert this image to anime style",
    ModelVariant.UNBLUR_ANYTHING: "Sharpen and unblur this image",
}


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit LoRA Collection model loader."""

    _VARIANTS = {
        ModelVariant.OBJECT_REMOVER_BBOX: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.GUIDED_HEAD_FACE_SWAP: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.BW_TO_TRUE_COLOR: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.ANIME: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.UNBLUR_ANYTHING: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.OBJECT_REMOVER_BBOX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_LORA_COLLECTION",
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
        """Load the Qwen-Image-Edit transformer with LoRA weights fused.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            lora_file = _LORA_FILES[self._variant]
            pipe.load_lora_weights(
                LORA_REPO,
                weight_name=lora_file,
            )
            pipe.fuse_lora()
            self._transformer = pipe.transformer
            self._transformer.eval()
            del pipe
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)

        return self._transformer

    def load_inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
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
