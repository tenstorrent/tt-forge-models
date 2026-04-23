# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Fill-dev GGUF model loader implementation for image inpainting.

This loader uses GGUF-quantized variants of the FLUX.1-Fill-dev model from
YarvixPA/FLUX.1-Fill-dev-GGUF. The GGUF transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file with a local config to bypass the gated
black-forest-labs/FLUX.1-Fill-dev repository.
"""

import json
import os
import tempfile
from typing import Optional

import torch
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

GGUF_REPO = "YarvixPA/FLUX.1-Fill-dev-GGUF"

# FLUX.1-Fill-dev transformer architecture constants
_CLIP_EMBED_DIM = 768
_T5_EMBED_DIM = 4096
_MAX_SEQ_LEN = 256
_VAE_SCALE_FACTOR = 8


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
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        from diffusers import GGUFQuantizationConfig
        from diffusers.models import FluxTransformer2DModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        # FLUX.1-Fill-dev is a gated repo; provide transformer config locally.
        # in_channels=384: 64 noisy + 64 masked-image + 256 packed-mask latents.
        # out_channels=64: proj_out still predicts the denoised 64-channel latent.
        flux_fill_config = {
            "_class_name": "FluxTransformer2DModel",
            "_diffusers_version": "0.33.1",
            "attention_head_dim": 128,
            "axes_dims_rope": [16, 56, 56],
            "guidance_embeds": True,
            "in_channels": 384,
            "out_channels": 64,
            "joint_attention_dim": _T5_EMBED_DIM,
            "num_attention_heads": 24,
            "num_layers": 19,
            "num_single_layers": 38,
            "patch_size": 1,
        }
        with tempfile.TemporaryDirectory() as config_dir:
            with open(os.path.join(config_dir, "config.json"), "w") as f:
                json.dump(flux_fill_config, f)
            self.transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                config=config_dir,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare synthetic sample inputs for the FLUX Fill transformer.

        The FLUX Fill transformer expects hidden_states that include the noisy latents
        concatenated with masked image latents and a packed mask along dim=2,
        resulting in in_channels of 384 (64 + 64 + 256).

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        height = 128
        width = 128

        # Latent dimensions after VAE encoding and FLUX packing
        latent_h = height // _VAE_SCALE_FACTOR
        latent_w = width // _VAE_SCALE_FACTOR
        seq_len = (latent_h // 2) * (latent_w // 2)

        # Noisy latents + masked image latents + packed mask concatenated
        noise_channels = 64
        masked_image_channels = 64
        mask_channels = 256
        total_channels = noise_channels + masked_image_channels + mask_channels
        hidden_states = torch.randn(batch_size, seq_len, total_channels, dtype=dtype)

        # Synthetic text embeddings (no tokenizer needed for compile-only)
        pooled_projections = torch.randn(batch_size, _CLIP_EMBED_DIM, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, _MAX_SEQ_LEN, _T5_EMBED_DIM, dtype=dtype
        )
        txt_ids = torch.zeros(_MAX_SEQ_LEN, 3, dtype=dtype)

        # Latent image position IDs
        latent_image_ids = torch.zeros(latent_h // 2, latent_w // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(latent_h // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(latent_w // 2)[None, :]
        )
        img_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": torch.tensor([3.5], dtype=dtype),
            "pooled_projections": pooled_projections,
            "encoder_hidden_states": encoder_hidden_states,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            "joint_attention_kwargs": {},
        }
