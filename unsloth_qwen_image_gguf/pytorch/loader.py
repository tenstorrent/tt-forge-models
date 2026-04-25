# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth Qwen-Image GGUF model loader implementation for image generation.

Repository:
- https://huggingface.co/unsloth/Qwen-Image-GGUF
"""
import json
import os
import tempfile
from typing import Optional

import torch
from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel
from huggingface_hub import hf_hub_download

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

GGUF_REPO = "unsloth/Qwen-Image-GGUF"

# QwenImage transformer config matching the Qwen/Qwen-Image model defaults.
# Provided locally to avoid downloading the full Qwen/Qwen-Image pipeline repo.
_TRANSFORMER_CONFIG = {
    "_class_name": "QwenImageTransformer2DModel",
    "patch_size": 2,
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 60,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 3584,
    "guidance_embeds": False,
    "axes_dims_rope": [16, 56, 56],
    "zero_cond_t": False,
    "use_additional_t_cond": False,
    "use_layer3d_rope": False,
}


class ModelVariant(StrEnum):
    """Available Unsloth Qwen-Image GGUF model variants."""

    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_K_M: "qwen-image-Q4_K_M.gguf",
    ModelVariant.Q8_0: "qwen-image-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Unsloth Qwen-Image GGUF model loader for image generation tasks."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Unsloth Qwen-Image GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(self, dtype: torch.dtype = torch.bfloat16):
        """Load the GGUF-quantized QwenImage transformer using a local config.

        A local config.json is written with the known transformer architecture
        parameters to avoid fetching from the full Qwen/Qwen-Image pipeline repo.
        """
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        with tempfile.TemporaryDirectory() as config_dir:
            with open(os.path.join(config_dir, "config.json"), "w") as f:
                json.dump(_TRANSFORMER_CONFIG, f)
            self._transformer = QwenImageTransformer2DModel.from_single_file(
                gguf_path,
                config=config_dir,
                quantization_config=quantization_config,
                torch_dtype=dtype,
            )
        return self._transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            self._load_transformer(dtype)
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self._transformer.config

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

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
