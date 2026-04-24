# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
z_image_turbo_mzbac_8bit model loader implementation.

Loads the mzbac/Z-Image-Turbo-8bit text-to-image pipeline, an 8-bit
group-wise affine quantized variant of Z-Image-Turbo distributed as a
full diffusers-format pipeline repo.

The transformer and text encoder use a custom uint32-packed int8 quantization
format that is not compatible with standard diffusers/transformers loading.
We load the transformer architecture from config (using ignore_mismatched_sizes
to skip quantized weights) and supply synthetic inputs for compile-only testing.

Available variants:
- Z_IMAGE_TURBO_MZBAC_8BIT: Full 8-bit quantized Z-Image-Turbo DiT transformer
"""

import glob
import json
import os
from typing import Any, Optional

import torch
from diffusers import ZImageTransformer2DModel
from huggingface_hub import snapshot_download

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

PIPELINE_REPO_ID = "mzbac/Z-Image-Turbo-8bit"

# Spatial resolution used for synthetic inputs (in latent space, pre-patch)
_LATENT_SPATIAL = 8  # latent h/w in patches after VAE downscale
_SEQ_LEN = 32  # synthetic text token count for compile
_CAP_FEAT_DIM = 2560  # matches config cap_feat_dim / Qwen3 hidden_size


class ModelVariant(StrEnum):
    """Available z_image_turbo_mzbac_8bit model variants."""

    Z_IMAGE_TURBO_MZBAC_8BIT = "Z-Image-Turbo-8bit"


def _create_diffusers_safetensors_index(directory: str) -> None:
    """Create diffusion_pytorch_model.safetensors.index.json if missing.

    Some repos use model-NNNNN-of-NNNNN.safetensors naming (transformers style)
    without a diffusers-style index file. This generates the index so diffusers
    can find the sharded weights.
    """
    index_path = os.path.join(
        directory, "diffusion_pytorch_model.safetensors.index.json"
    )
    if os.path.exists(index_path):
        return
    shard_files = sorted(glob.glob(os.path.join(directory, "model-*-of-*.safetensors")))
    if not shard_files:
        return
    from safetensors import safe_open

    weight_map = {}
    total_size = 0
    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                weight_map[key] = shard_name
                total_size += tensor.numel() * tensor.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(index_path, "w") as fp:
        json.dump(index, fp, indent=2)


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
        self._transformer = None

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

    def _load_transformer(self, dtype: torch.dtype = torch.bfloat16):
        """Load the ZImageTransformer2DModel from the 8-bit quantized checkpoint.

        The checkpoint uses a custom uint32-packed int8 quantization that is
        incompatible with standard diffusers loading. We load only the model
        architecture from config, skipping mismatched quantized weight tensors,
        which is sufficient for compile-only evaluation.
        """
        local_dir = snapshot_download(PIPELINE_REPO_ID)
        transformer_dir = os.path.join(local_dir, "transformer")
        _create_diffusers_safetensors_index(transformer_dir)
        self._transformer = ZImageTransformer2DModel.from_pretrained(
            transformer_dir,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )
        # Materialize any meta tensors left by mismatched-size skips
        for name, param in self._transformer.named_parameters():
            if param.is_meta:
                new_data = torch.empty(param.shape, dtype=param.dtype, device="cpu")
                torch.nn.init.normal_(new_data)
                param_ref = dict(self._transformer.named_parameters())[name]
                param_ref.data = new_data
        for name, buf in self._transformer.named_buffers():
            if buf.is_meta:
                new_data = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                buf_ref = dict(self._transformer.named_buffers())[name]
                buf_ref.data = new_data
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the DiT transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare synthetic transformer inputs for compile-only evaluation.

        The forward signature is:
            forward(x, t, cap_feats, ...)
        where x is a list of [C, F, H, W] tensors, t is timestep, cap_feats
        is a list of [T, D] text-embedding tensors (one per image in the batch).
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        if self._transformer is None:
            self._load_transformer(dtype)

        in_channels = self._transformer.in_channels  # 16
        h = w = _LATENT_SPATIAL

        # x: list of [C, F, H, W] tensors (one per image)
        latent = torch.randn(in_channels, 1, h, w, dtype=dtype)
        x = [latent]

        # t: batch of timesteps in [0, 1]
        t = torch.tensor([0.5], dtype=dtype)

        # cap_feats: list of [T, D] text-embedding tensors (one per image)
        cap_feat = torch.randn(_SEQ_LEN, _CAP_FEAT_DIM, dtype=dtype)
        cap_feats = [cap_feat]

        return [x, t, cap_feats]
