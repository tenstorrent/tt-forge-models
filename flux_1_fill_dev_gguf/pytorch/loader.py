# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Fill-dev GGUF model loader implementation for image inpainting.

This loader uses GGUF-quantized variants of the FLUX.1-Fill-dev model from
YarvixPA/FLUX.1-Fill-dev-GGUF. The GGUF transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file with a local path (via hf_hub_download)
and synthetic inputs, avoiding any dependency on the gated base repo.
"""

import json
import os
import tempfile
from typing import Optional

import torch
import torch._C
from diffusers import GGUFQuantizationConfig
from diffusers.models import FluxTransformer2DModel
from diffusers.quantizers.gguf.utils import GGUFParameter
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


def _patched_as_tensor(self):
    # GGUFParameter.__torch_function__ causes infinite recursion when _make_subclass
    # is called on a GGUFParameter instance; DisableTorchFunctionSubclass breaks the loop.
    with torch._C.DisableTorchFunctionSubclass():
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)


GGUFParameter.as_tensor = _patched_as_tensor

GGUF_REPO = "YarvixPA/FLUX.1-Fill-dev-GGUF"

# FLUX.1-Fill-dev transformer architecture config.
# in_channels=384: fill model concatenates noisy latents (64), masked-image
#   latents (64), and packed mask (256) along the channel dim before packing.
# out_channels=64: proj_out predicts only the noise channels, not the full
#   concatenated input. Without this, diffusers defaults out_channels=in_channels
#   and expects proj_out.weight=[384,3072] but the GGUF has [64,3072].
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
    "in_channels": 384,
    "out_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


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

    def _make_local_config_dir(self):
        """Write transformer/config.json locally to avoid fetching from the gated base repo."""
        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(_TRANSFORMER_CONFIG, f)
        return config_dir

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX Fill transformer.

        Uses hf_hub_download for a local path instead of a direct HTTPS URL because
        diffusers 0.37.1 _extract_repo_id_and_weights_name only strips blob/main/ (not
        resolve/main/), so weights_name ends up as "resolve/main/file.gguf" and the
        subsequent HF hub download appends another resolve/main/, yielding a 404.

        Returns:
            torch.nn.Module: The FLUX Fill transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.transformer is None:
            gguf_file = _GGUF_FILES[self._variant]
            model_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
            quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
            config_dir = self._make_local_config_dir()

            self.transformer = FluxTransformer2DModel.from_single_file(
                model_path,
                config=config_dir,
                subfolder="transformer",
                quantization_config=quantization_config,
                torch_dtype=dtype,
            )
        elif dtype_override is not None:
            self.transformer = self.transformer.to(dtype=dtype_override)
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare synthetic inputs for the FLUX Fill transformer.

        Inputs are derived from the inline _TRANSFORMER_CONFIG rather than running
        real text encoders, so the loader does not require access to the gated
        black-forest-labs/FLUX.1-Fill-dev base repo.

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        config = self.transformer.config
        height = 128
        width = 128
        vae_scale_factor = 8
        max_sequence_length = 256

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2
        seq_len = h_packed * w_packed

        # Noisy latents (64 ch) + masked-image latents (64 ch) + packed mask (256 ch)
        noise_channels = 64
        masked_image_channels = 64
        mask_channels = 256
        latents = torch.randn(batch_size, seq_len, noise_channels, dtype=dtype)
        masked_image_latents = torch.randn(
            batch_size, seq_len, masked_image_channels + mask_channels, dtype=dtype
        )
        hidden_states = torch.cat((latents, masked_image_latents), dim=2)

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, config.joint_attention_dim, dtype=dtype
        )
        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)
        guidance = torch.tensor([3.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "timestep": torch.tensor([1.0], dtype=dtype).expand(batch_size),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
