# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SPAR3D model loader implementation for image-to-3D mesh reconstruction.

Loads the TwoStreamInterleaveTransformer (triplane backbone) from Stability AI's
Stable Point Aware 3D pipeline, which fuses image tokens with learnable triplane
latents to produce triplane features used for mesh reconstruction.

Requires the stable-point-aware-3d repository to be cloned at /tmp/spar3d_repo.
"""
import os
import sys
import types
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file

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

SPAR3D_REPO_PATH = "/tmp/spar3d_repo"


def _ensure_spar3d_importable():
    """Ensure the stable-point-aware-3d repo is cloned and importable.

    Registers stub packages so we can import the backbone transformer directly
    without triggering spar3d/system.py, which requires optional CUDA-only
    extensions (texture_baker, uv_unwrapper).
    """
    if "spar3d" not in sys.modules:
        if not os.path.isdir(SPAR3D_REPO_PATH):
            import subprocess

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/Stability-AI/stable-point-aware-3d.git",
                    SPAR3D_REPO_PATH,
                ]
            )

        if SPAR3D_REPO_PATH not in sys.path:
            sys.path.insert(0, SPAR3D_REPO_PATH)

        base = os.path.join(SPAR3D_REPO_PATH, "spar3d")
        for name, subpath in [
            ("spar3d", ""),
            ("spar3d.models", "models"),
            ("spar3d.models.transformers", os.path.join("models", "transformers")),
        ]:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(base, subpath)]
            sys.modules[name] = mod


class ModelVariant(StrEnum):
    """Available SPAR3D model variants."""

    TRIPLANE_BACKBONE = "Triplane_Backbone"


class ModelLoader(ForgeModel):
    """SPAR3D model loader for the TwoStreamInterleaveTransformer backbone."""

    _VARIANTS = {
        ModelVariant.TRIPLANE_BACKBONE: ModelConfig(
            pretrained_model_name="stabilityai/stable-point-aware-3d",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRIPLANE_BACKBONE

    _CONFIG_NAME = "config.yaml"
    _WEIGHT_NAME = "model.safetensors"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SPAR3D",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SPAR3D triplane backbone transformer.

        Returns:
            torch.nn.Module: The TwoStreamInterleaveTransformer backbone.
        """
        _ensure_spar3d_importable()
        from spar3d.models.transformers.backbone import TwoStreamInterleaveTransformer

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id, self._CONFIG_NAME)
        weights_path = hf_hub_download(repo_id, self._WEIGHT_NAME)

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        backbone_cfg = cfg.backbone

        model = TwoStreamInterleaveTransformer(backbone_cfg)

        state_dict = load_file(weights_path)
        backbone_state = {
            k.removeprefix("backbone."): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        model.load_state_dict(backbone_state, strict=False)
        model.eval()

        # Capture architecture constants for input construction.
        self._triplane_channels = backbone_cfg.get("raw_triplane_channels", 1024)
        self._image_channels = backbone_cfg.get("raw_image_channels", 1024)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the TwoStreamInterleaveTransformer.

        Returns:
            dict: Input tensors (hidden_states, encoder_hidden_states) for the
                  model forward pass.
        """
        dtype = dtype_override or torch.float32

        triplane_channels = getattr(self, "_triplane_channels", 1024)
        image_channels = getattr(self, "_image_channels", 1024)

        # Triplane resolution 96 -> 3 planes * 96 * 96 = 27648 tokens.
        num_triplane_tokens = 3 * 96 * 96
        # DINOv2 ViT-L/14 @ 518px: (518/14)^2 + 1 = 1370 tokens.
        num_image_tokens = 1370

        hidden_states = torch.randn(
            batch_size,
            triplane_channels,
            num_triplane_tokens,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            num_image_tokens,
            image_channels,
            dtype=dtype,
        )

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
        }
