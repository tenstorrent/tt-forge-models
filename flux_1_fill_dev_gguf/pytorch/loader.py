# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Fill-dev GGUF model loader implementation for image inpainting.

This loader uses GGUF-quantized variants of the FLUX.1-Fill-dev model from
YarvixPA/FLUX.1-Fill-dev-GGUF. The GGUF transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file and plugged into a FluxFillPipeline
built from the original black-forest-labs/FLUX.1-Fill-dev repository.
"""

from typing import Optional

import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear

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

GGUF_REPO = "YarvixPA/FLUX.1-Fill-dev-GGUF"
BASE_REPO = "black-forest-labs/FLUX.1-Fill-dev"


class ModelVariant(StrEnum):
    """Available FLUX.1-Fill-dev GGUF quantization variants."""

    Q3_K_S = "Q3_K_S"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K_S = "Q4_K_S"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K_S = "Q5_K_S"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q3_K_S: "flux1-fill-dev-Q3_K_S.gguf",
    ModelVariant.Q4_0: "flux1-fill-dev-Q4_0.gguf",
    ModelVariant.Q4_1: "flux1-fill-dev-Q4_1.gguf",
    ModelVariant.Q4_K_S: "flux1-fill-dev-Q4_K_S.gguf",
    ModelVariant.Q5_0: "flux1-fill-dev-Q5_0.gguf",
    ModelVariant.Q5_1: "flux1-fill-dev-Q5_1.gguf",
    ModelVariant.Q5_K_S: "flux1-fill-dev-Q5_K_S.gguf",
    ModelVariant.Q6_K: "flux1-fill-dev-Q6_K.gguf",
    ModelVariant.Q8_0: "flux1-fill-dev-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX.1-Fill-dev GGUF model loader for image inpainting."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-Fill-dev GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX Fill transformer.

        Returns:
            torch.nn.Module: The FLUX Fill transformer model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        # Use blob/main URL: diffusers strips "blob/main/" from the filename;
        # resolve/main causes the path to be doubled → 404.
        self.transformer = FluxTransformer2DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
            config=BASE_REPO,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )
        # GGUFParameter.__torch_function__ recurses infinitely under TorchDynamo.
        # Dequantize to plain tensors before compilation reaches the model.
        # F16-stored GGUF tensors dequantize to float16, not compute_dtype, so
        # cast via torch.nn.Module.to() directly — diffusers ModelMixin.to()
        # raises ValueError on quantized models even after dequantization.
        _dequantize_gguf_and_restore_linear(self.transformer)
        torch.nn.Module.to(self.transformer, compute_dtype)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX Fill transformer.

        The FLUX Fill transformer expects hidden_states that include the noisy latents
        concatenated with masked image latents and a packed mask along dim=2,
        resulting in in_channels of 384 (64 + 64 + 256).

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        max_sequence_length = 256
        height = 128
        width = 128
        vae_scale_factor = 8  # standard FLUX VAE scale factor

        # Latent dimensions
        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))
        seq_len = (height_latent // 2) * (width_latent // 2)

        # Noisy latents (packed format)
        noise_channels = 64
        latents = torch.randn(
            batch_size, seq_len, noise_channels, dtype=dtype
        )

        # Masked image latents (VAE-encoded masked image + packed mask)
        masked_image_channels = 64
        mask_channels = 256
        masked_image_latents = torch.randn(
            batch_size,
            seq_len,
            masked_image_channels + mask_channels,
            dtype=dtype,
        )

        # Concatenate noisy latents with masked image latents along channel dim
        hidden_states = torch.cat((latents, masked_image_latents), dim=2)

        # Latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Synthetic text embeddings (correct shapes, no pipeline required)
        # CLIP pooler output: (batch, 768)
        pooled_prompt_embeds = torch.randn(batch_size, 768, dtype=dtype)
        # T5 encoder output: (batch, max_sequence_length, 4096)
        prompt_embeds = torch.randn(batch_size, max_sequence_length, 4096, dtype=dtype)
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        # FLUX.1-Fill-dev uses classifier-free guidance
        guidance = torch.tensor([3.5], dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
