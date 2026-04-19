# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth Qwen-Image GGUF model loader implementation for text-to-image generation.

Repository:
- https://huggingface.co/unsloth/Qwen-Image-GGUF
"""
import torch
from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel
from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
from huggingface_hub import hf_hub_download
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

REPO_ID = "unsloth/Qwen-Image-GGUF"
CONFIG_REPO = "Qwen/Qwen-Image-2512"


class ModelVariant(StrEnum):
    """Available Unsloth Qwen-Image GGUF model variants."""

    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """Unsloth Qwen-Image GGUF model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "qwen-image-Q4_K_M.gguf",
        ModelVariant.Q8_0: "qwen-image-Q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

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

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = self._GGUF_FILES[self._variant]
        model_path = hf_hub_download(repo_id=REPO_ID, filename=gguf_file)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)

        self.transformer = QwenImageTransformer2DModel.from_single_file(
            model_path,
            config=CONFIG_REPO,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )

        self._dequantize_model(dtype)
        return self.transformer

    def _dequantize_model(self, dtype):
        for name, param in list(self.transformer.named_parameters()):
            if isinstance(param, GGUFParameter):
                parts = name.split(".")
                module = self.transformer
                for part in parts[:-1]:
                    module = getattr(module, part)
                dequantized = dequantize_gguf_tensor(param).to(dtype)
                setattr(
                    module,
                    parts[-1],
                    torch.nn.Parameter(dequantized, requires_grad=False),
                )

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Image dimensions
        height = 128
        width = 128
        patch_size = config.patch_size
        in_channels = config.in_channels

        # Compute latent sequence length: (H / patch) * (W / patch)
        h_patches = height // patch_size
        w_patches = width // patch_size
        image_seq_len = h_patches * w_patches

        # Hidden states: (batch, image_seq_len, in_channels)
        hidden_states = torch.randn(batch_size, image_seq_len, in_channels, dtype=dtype)

        # Encoder hidden states (text embeddings): (batch, text_seq_len, joint_attention_dim)
        text_seq_len = 128
        joint_attention_dim = config.joint_attention_dim
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        # Encoder hidden states mask: (batch, text_seq_len)
        encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        # Image shapes for RoPE: list of (frames, height, width)
        img_shapes = [(1, h_patches, w_patches)] * batch_size

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }

        return inputs
