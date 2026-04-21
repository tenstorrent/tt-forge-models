# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
z_image_turbo_mzbac_8bit model loader implementation.

Loads the mzbac/Z-Image-Turbo-8bit text-to-image pipeline, an 8-bit
group-wise affine quantized variant of Z-Image-Turbo distributed as a
full diffusers-format pipeline repo.

Available variants:
- Z_IMAGE_TURBO_MZBAC_8BIT: Full 8-bit quantized Z-Image-Turbo DiT transformer
"""

import os
from typing import Any, Optional

import numpy as np
import torch
from diffusers import ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open

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

PIPELINE_REPO_ID = "mzbac/Z-Image-Turbo-8bit"
QUANT_GROUP_SIZE = 32


def _dequantize_packed_int8(
    weight_uint32: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Each uint32 stores 4 int8 values in little-endian byte order.
    # Unpack via numpy for reliable byte-level reinterpretation.
    out_dim, ncols_packed = weight_uint32.shape
    in_dim = ncols_packed * 4
    groups = in_dim // QUANT_GROUP_SIZE

    weight_int8 = torch.from_numpy(weight_uint32.numpy().view(np.int8)).reshape(
        out_dim, in_dim
    )
    weight_grouped = weight_int8.float().reshape(out_dim, groups, QUANT_GROUP_SIZE)
    dequantized = (
        weight_grouped * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    ).reshape(out_dim, in_dim)
    return dequantized.to(dtype)


def _load_dequantized_state_dict(
    transformer_dir: str, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    shard_files = sorted(
        f
        for f in os.listdir(transformer_dir)
        if f.endswith(".safetensors") and "model" in f
    )

    raw_state: dict[str, torch.Tensor] = {}
    for shard_file in shard_files:
        with safe_open(
            os.path.join(transformer_dir, shard_file), framework="pt", device="cpu"
        ) as f:
            for k in f.keys():
                raw_state[k] = f.get_tensor(k)

    quant_bases = {
        k.rsplit(".", 1)[0] for k, v in raw_state.items() if v.dtype == torch.uint32
    }

    state_dict: dict[str, torch.Tensor] = {}

    for base in quant_bases:
        dequantized = _dequantize_packed_int8(
            raw_state[f"{base}.weight"],
            raw_state[f"{base}.scales"],
            raw_state[f"{base}.biases"],
            dtype,
        )
        state_dict[f"{base}.weight"] = dequantized

    skip_keys = {f"{b}.scales" for b in quant_bases} | {
        f"{b}.biases" for b in quant_bases
    }
    for key, tensor in raw_state.items():
        if key in skip_keys or tensor.dtype == torch.uint32:
            continue
        state_dict[key] = tensor.to(dtype) if tensor.is_floating_point() else tensor

    return state_dict


class ModelVariant(StrEnum):
    """Available z_image_turbo_mzbac_8bit model variants."""

    Z_IMAGE_TURBO_MZBAC_8BIT = "Z-Image-Turbo-8bit"


class ModelLoader(ForgeModel):
    """z_image_turbo_mzbac_8bit model loader."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE_TURBO_MZBAC_8BIT: ModelConfig(
            pretrained_model_name=PIPELINE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_TURBO_MZBAC_8BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_TURBO_MZBAC_8BIT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(self, dtype: torch.dtype) -> ZImageTransformer2DModel:
        # The repo's model_index.json references the old QwenImageTransformer2DModel
        # class name, and the weights use a custom packed int8 quantization (4 int8
        # values per uint32) with per-group scales/biases that standard from_pretrained
        # cannot handle. Load the config, build the model, then dequantize and inject
        # the weights manually.
        snapshot_dir = snapshot_download(PIPELINE_REPO_ID)
        transformer_dir = os.path.join(snapshot_dir, "transformer")

        config, _ = ZImageTransformer2DModel.load_config(
            transformer_dir, return_unused_kwargs=True
        )
        model = ZImageTransformer2DModel.from_config(config)

        state_dict = _load_dequantized_state_dict(transformer_dir, dtype)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                f"[z_image_turbo] Missing keys when loading transformer: {len(missing)}"
            )
        if unexpected:
            print(
                f"[z_image_turbo] Unexpected keys when loading transformer: {len(unexpected)}"
            )

        return model.to(dtype)

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the mzbac 8-bit quantized Z-Image-Turbo pipeline."""
        transformer = self._load_transformer(dtype)
        self._pipe = ZImagePipeline.from_pretrained(
            PIPELINE_REPO_ID,
            transformer=transformer,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the DiT transformer from the quantized pipeline."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare transformer inputs (latents, timestep, prompt_embeds)."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A photo of an astronaut riding a horse on mars"

        if self._pipe is None:
            self._load_pipeline(dtype)

        prompt_embeds, _ = self._pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        num_channels_latents = self._pipe.transformer.config.in_channels
        vae_scale = self._pipe.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
