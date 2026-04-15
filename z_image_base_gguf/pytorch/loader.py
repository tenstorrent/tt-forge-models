# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image Base GGUF model loader implementation for text-to-image generation.

Loads the GGUF-quantized DiT transformer from babakarto/z-image-base-gguf.
"""

import math

import torch
from diffusers import GGUFQuantizationConfig, ZImagePipeline
from diffusers.models import ZImageTransformer2DModel
from diffusers.quantizers.gguf.gguf_quantizer import GGUFQuantizer
from diffusers.quantizers.gguf.utils import (
    _quant_shape_from_byte_shape,
    GGML_QUANT_SIZES,
)
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

REPO_ID = "babakarto/z-image-base-gguf"
PIPELINE_REPO_ID = "Tongyi-MAI/Z-Image"


def _lenient_check_quantized_param_shape(self, param_name, current_param, loaded_param):
    """Allow shape mismatches when total element counts match (e.g. (3840,) vs (1, 3840))."""
    loaded_param_shape = loaded_param.shape
    current_param_shape = current_param.shape
    quant_type = loaded_param.quant_type
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    inferred_shape = _quant_shape_from_byte_shape(
        loaded_param_shape, type_size, block_size
    )
    if inferred_shape != current_param_shape:
        if math.prod(inferred_shape) != math.prod(current_param_shape):
            raise ValueError(
                f"{param_name} has an expected quantized shape of: {inferred_shape}, "
                f"but received shape: {loaded_param_shape}"
            )
    return True


GGUFQuantizer.check_quantized_param_shape = _lenient_check_quantized_param_shape


class ModelVariant(StrEnum):
    """Available Z-Image Base GGUF model variants."""

    BASE_Q8_0 = "Base_Q8_0"


class ModelLoader(ForgeModel):
    """Z-Image Base GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.BASE_Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_Q8_0

    GGUF_FILE = "z_image_base_Q8_0.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Z-Image Base GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        repo_id = self._variant_config.pretrained_model_name
        self.transformer = ZImageTransformer2DModel.from_single_file(
            f"https://huggingface.co/{repo_id}/{self.GGUF_FILE}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Image dimensions
        height = 128
        width = 128

        # Z-Image uses patch_size=2 and in_channels=16 (Lumina2/DiT architecture)
        in_channels = config.in_channels
        patch_size = config.all_patch_size[0]
        latent_h = height // (patch_size * 8)
        latent_w = width // (patch_size * 8)

        # Prepare latent input: list of tensors [1, channels, 1, H, W] per batch item
        latents = torch.randn(
            batch_size, in_channels, 1, latent_h, latent_w, dtype=dtype
        )
        latent_input_list = list(latents.unbind(dim=0))

        # Timestep (normalized: 0 = fully denoised, 1 = fully noised)
        timestep = torch.tensor([0.5], dtype=dtype)

        # Caption/prompt embeddings
        cap_feat_dim = config.cap_feat_dim
        seq_len = 32
        prompt_embeds = torch.randn(batch_size, seq_len, cap_feat_dim, dtype=dtype)

        return [latent_input_list, timestep, prompt_embeds]
