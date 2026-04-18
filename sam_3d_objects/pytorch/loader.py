# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAM-3D-Objects model loader implementation for 3D object reconstruction.

Loads the SparseStructureEncoder from the SAM-3D pipeline, which encodes
3D input into a latent representation for downstream 3D reconstruction.

Requires the sam-3d-objects repository to be cloned at /tmp/sam3d_objects_repo.
"""
import importlib.util
import os
import sys

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
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

SAM3D_REPO_PATH = "/tmp/sam3d_objects_repo"


def _ensure_sam3d_importable():
    """Ensure the sam-3d-objects repo is cloned and importable."""
    if not os.path.isdir(SAM3D_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--recurse-submodules",
                "https://github.com/facebookresearch/sam-3d-objects.git",
                SAM3D_REPO_PATH,
            ]
        )
    if SAM3D_REPO_PATH not in sys.path:
        sys.path.insert(0, SAM3D_REPO_PATH)


def _load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_sparse_structure_encoder():
    base = os.path.join(
        SAM3D_REPO_PATH, "sam3d_objects", "model", "backbone", "tdfy_dit"
    )
    modules_dir = os.path.join(base, "modules")
    norm_mod = _load_module_from_file(
        "sam3d_objects.model.backbone.tdfy_dit.modules.norm",
        os.path.join(modules_dir, "norm.py"),
    )
    # Patch norm classes to remove mixed-precision .float() calls that conflict
    # with uniform bf16 dtype required by the TT compiler.
    norm_mod.LayerNorm32.forward = lambda self, x: torch.nn.LayerNorm.forward(self, x)
    norm_mod.GroupNorm32.forward = lambda self, x: torch.nn.GroupNorm.forward(self, x)

    def _channel_ln_forward(self, x):
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = torch.nn.LayerNorm.forward(self, x)
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()
        return x

    norm_mod.ChannelLayerNorm32.forward = _channel_ln_forward
    _load_module_from_file(
        "sam3d_objects.model.backbone.tdfy_dit.modules.spatial",
        os.path.join(modules_dir, "spatial.py"),
    )
    _load_module_from_file(
        "sam3d_objects.model.backbone.tdfy_dit.modules.utils",
        os.path.join(modules_dir, "utils.py"),
    )
    vae_mod = _load_module_from_file(
        "sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae",
        os.path.join(base, "models", "sparse_structure_vae.py"),
    )
    return vae_mod.SparseStructureEncoder


class _SparseStructureEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, sample_posterior=False, return_raw=False):
        h = self.encoder.input_layer(x)
        for block in self.encoder.blocks:
            h = block(h)
        h = self.encoder.middle_block(h)
        h = self.encoder.out_layer(h)
        mean, logvar = h.chunk(2, dim=1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        if return_raw:
            return z, mean, logvar
        return z


class ModelVariant(StrEnum):
    """Available SAM-3D-Objects model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """SAM-3D-Objects model loader for the SparseStructureEncoder."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="facebook/sam-3d-objects",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    _CKPT_NAME = "checkpoints/ss_encoder"
    _RESOLUTION = 16
    _IN_CHANNELS = 8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SAM-3D-Objects",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    _DEFAULT_MODEL_ARGS = {
        "in_channels": 8,
        "latent_channels": 8,
        "num_res_blocks": 2,
        "channels": [32, 64, 128, 256],
        "num_res_blocks_middle": 2,
        "norm_type": "layer",
        "use_fp16": False,
    }

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_sam3d_importable()
        os.environ["LIDRA_SKIP_INIT"] = "1"
        SparseStructureEncoder = _import_sparse_structure_encoder()

        if os.environ.get("TT_RANDOM_WEIGHTS") == "1":
            model = SparseStructureEncoder(**self._DEFAULT_MODEL_ARGS)
        else:
            repo_id = self._variant_config.pretrained_model_name
            config_path = hf_hub_download(repo_id, f"{self._CKPT_NAME}.yaml")
            weights_path = hf_hub_download(repo_id, f"{self._CKPT_NAME}.safetensors")

            with open(config_path) as f:
                config = yaml.safe_load(f)

            model_args = {k: v for k, v in config.items() if not k.startswith("_")}
            model = SparseStructureEncoder(**model_args)

            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict)

        model = _SparseStructureEncoderWrapper(model)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the SparseStructureEncoder.

        Returns:
            torch.Tensor: A 3D volume tensor [B, C, D, H, W].
        """
        dtype = dtype_override or torch.float32

        # x: 3D volume input [B, C, D, H, W]
        x = torch.randn(
            batch_size,
            self._IN_CHANNELS,
            self._RESOLUTION,
            self._RESOLUTION,
            self._RESOLUTION,
            dtype=dtype,
        )

        return x
