# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
calcuis/qwen-image-gguf model loader implementation for text-to-image generation.
"""

from typing import Optional

import torch
from diffusers import (
    GGUFQuantizationConfig,
    QwenImageTransformer2DModel,
)
from huggingface_hub import hf_hub_download

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen-Image GGUF model variants."""

    QWEN_IMAGE_Q4_K_M = "qwen-image-q4_k_m"


class ModelLoader(ForgeModel):
    """calcuis/qwen-image-gguf model loader implementation."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_Q4_K_M: ModelConfig(
            pretrained_model_name="calcuis/qwen-image-gguf",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.QWEN_IMAGE_Q4_K_M: "qwen-image-q4_k_m.gguf",
    }

    _PIPELINE_REPO = "callgg/qi-decoder"

    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen-Image-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = self._GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=gguf_file,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            config=self._PIPELINE_REPO,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
            torch_dtype=compute_dtype,
        )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        height = 128
        width = 128
        patch_size = config.patch_size
        in_channels = config.in_channels

        h_patches = height // patch_size
        w_patches = width // patch_size
        image_seq_len = h_patches * w_patches

        hidden_states = torch.randn(batch_size, image_seq_len, in_channels, dtype=dtype)

        text_seq_len = 128
        joint_attention_dim = config.joint_attention_dim
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        img_shapes = [(1, h_patches, w_patches)] * batch_size

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }

        return inputs
