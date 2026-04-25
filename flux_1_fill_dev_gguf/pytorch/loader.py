# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Fill-dev GGUF model loader implementation for image inpainting.

This loader uses GGUF-quantized variants of the FLUX.1-Fill-dev model from
YarvixPA/FLUX.1-Fill-dev-GGUF. The GGUF transformer is loaded standalone via
diffusers' FluxTransformer2DModel.from_single_file. Inputs are constructed
from synthetic tensors with the correct shapes to avoid requiring the gated
black-forest-labs/FLUX.1-Fill-dev base repository.
"""

import importlib.util
import os
import sys
from typing import Optional

import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

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

# Local config dir avoids fetching from the gated black-forest-labs/FLUX.1-Fill-dev repo.
_TRANSFORMER_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "transformer_config")

# FLUX text encoder dimensions (CLIP-ViT-L/14 and T5-XXL)
_CLIP_POOLED_DIM = 768
_T5_SEQ_DIM = 4096
_VAE_SCALE_FACTOR = 16


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


def _refresh_diffusers_gguf_cache():
    """Fix diffusers module-level gguf availability cache after dynamic installation.

    diffusers caches gguf availability at import time. When RequirementsManager
    installs gguf dynamically, three things need patching:
    1. import_utils._gguf_available / _gguf_version - used by is_gguf_available()
    2. gguf_quantizer module-level names (_replace_with_gguf_linear etc.) that
       were skipped by the is_gguf_available() guard at module load time.
    """
    import importlib.metadata

    if importlib.util.find_spec("gguf") is not None:
        diu = sys.modules.get("diffusers.utils.import_utils")
        if diu is not None and not diu._gguf_available:
            diu._gguf_available = True
            try:
                diu._gguf_version = importlib.metadata.version("gguf")
            except importlib.metadata.PackageNotFoundError:
                diu._gguf_version = "0.10.0"

            # gguf_quantizer skips torch and gguf utility imports when gguf is absent
            # at module load time. Inject them now that gguf is installed.
            gguf_qmod = sys.modules.get("diffusers.quantizers.gguf.gguf_quantizer")
            if gguf_qmod is not None and not hasattr(
                gguf_qmod, "_replace_with_gguf_linear"
            ):
                import torch

                from diffusers.quantizers.gguf.utils import (
                    GGML_QUANT_SIZES,
                    GGUFParameter,
                    _dequantize_gguf_and_restore_linear,
                    _quant_shape_from_byte_shape,
                    _replace_with_gguf_linear,
                )

                gguf_qmod.torch = torch
                gguf_qmod.GGML_QUANT_SIZES = GGML_QUANT_SIZES
                gguf_qmod.GGUFParameter = GGUFParameter
                gguf_qmod._dequantize_gguf_and_restore_linear = (
                    _dequantize_gguf_and_restore_linear
                )
                gguf_qmod._quant_shape_from_byte_shape = _quant_shape_from_byte_shape
                gguf_qmod._replace_with_gguf_linear = _replace_with_gguf_linear


class ModelLoader(ForgeModel):
    """FLUX.1-Fill-dev GGUF model loader for image inpainting."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

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
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            _refresh_diffusers_gguf_cache()

            gguf_file = _GGUF_FILES[self._variant]
            quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)

            self._transformer = FluxTransformer2DModel.from_single_file(
                f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
                config=_TRANSFORMER_CONFIG_DIR,
                quantization_config=quantization_config,
                torch_dtype=dtype,
            )
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)

        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX Fill transformer.

        The FLUX Fill transformer expects hidden_states that include the noisy latents
        concatenated with masked image latents and a packed mask along dim=2,
        resulting in in_channels of 384 (64 + 64 + 256).

        Inputs are constructed from synthetic tensors using known FLUX architecture
        dimensions to avoid requiring the gated base repository.

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        max_sequence_length = 256
        height = 128
        width = 128
        num_images_per_prompt = 1

        # CLIP pooled text embeddings (CLIP-ViT-L/14: 768-dim)
        pooled_prompt_embeds = torch.zeros(
            batch_size * num_images_per_prompt, _CLIP_POOLED_DIM, dtype=dtype
        )

        # T5 sequence embeddings (T5-XXL: 4096-dim)
        prompt_embeds = torch.zeros(
            batch_size * num_images_per_prompt,
            max_sequence_length,
            _T5_SEQ_DIM,
            dtype=dtype,
        )

        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        # Latent dimensions (FLUX VAE scale factor is 16)
        height_latent = 2 * (int(height) // (_VAE_SCALE_FACTOR * 2))
        width_latent = 2 * (int(width) // (_VAE_SCALE_FACTOR * 2))
        seq_len = (height_latent // 2) * (width_latent // 2)

        # Noisy latents (packed format)
        noise_channels = 64
        latents = torch.randn(
            batch_size * num_images_per_prompt, seq_len, noise_channels, dtype=dtype
        )

        # Masked image latents (VAE-encoded masked image + packed mask)
        masked_image_channels = 64
        mask_channels = 256
        masked_image_latents = torch.randn(
            batch_size * num_images_per_prompt,
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
