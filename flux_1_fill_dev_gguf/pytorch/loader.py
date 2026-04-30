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

import json
import os
import tempfile
from typing import Optional

import torch
import torch._C
from diffusers import FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
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
BASE_REPO = "black-forest-labs/FLUX.1-Fill-dev"

# FLUX.1-Fill-dev transformer architecture config; in_channels=384 because
# the fill model concatenates noisy latents (64), masked-image latents (64),
# and packed mask (256) along the channel dim before packing.
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
    "in_channels": 384,
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
        self.pipe = None

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

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16):
        """Load the FluxFillPipeline with a GGUF-quantized transformer."""
        gguf_file = _GGUF_FILES[self._variant]
        # hf_hub_download returns a local path; from_single_file with a full
        # HTTPS resolve/main URL is mishandled by diffusers 0.37.1 — it fails to
        # strip resolve/main/ from the path, producing a doubled prefix in the
        # final download URL (404).
        model_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        config_dir = self._make_local_config_dir()

        transformer = FluxTransformer2DModel.from_single_file(
            model_path,
            config=config_dir,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )

        self.pipe = FluxFillPipeline.from_pretrained(
            BASE_REPO,
            transformer=transformer,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX Fill transformer.

        Returns:
            torch.nn.Module: The FLUX Fill transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipe is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype=dtype_override)
        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX Fill transformer.

        The FLUX Fill transformer expects hidden_states that include the noisy latents
        concatenated with masked image latents and a packed mask along dim=2,
        resulting in in_channels of 384 (64 + 64 + 256).

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipe is None:
            self._load_pipeline(dtype)

        max_sequence_length = 256
        prompt = "A cat sitting on a windowsill"
        height = 128
        width = 128
        num_images_per_prompt = 1

        # CLIP text encoding
        text_inputs_clip = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.pipe.text_encoder(
            text_inputs_clip.input_ids, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        # T5 text encoding
        text_inputs_t5 = self.pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.pipe.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Latent dimensions
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))
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
