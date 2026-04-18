# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MASt3R (Matching And Stereo 3D Reconstruction) model loader implementation.

Requires the mast3r repository to be cloned at /tmp/mast3r_repo.
"""
import os
import sys

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel

MAST3R_REPO_PATH = "/tmp/mast3r_repo"
IMG_HEIGHT = 384
IMG_WIDTH = 512


def _ensure_mast3r_importable():
    """Ensure the mast3r repo is cloned and importable."""
    if not os.path.isdir(MAST3R_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--recurse-submodules",
                "https://github.com/naver/mast3r.git",
                MAST3R_REPO_PATH,
            ]
        )

    if MAST3R_REPO_PATH not in sys.path:
        sys.path.insert(0, MAST3R_REPO_PATH)
    dust3r_path = os.path.join(MAST3R_REPO_PATH, "dust3r")
    if dust3r_path not in sys.path:
        sys.path.insert(0, dust3r_path)


def _patch_dust3r_for_dynamo():
    """Patch dust3r functions for torch dynamo compatibility.

    The model uses data-dependent control flow (assert with allclose,
    int() on tensors, is_symmetrized check) that breaks tracing.
    We replace these with dynamo-friendly versions using fixed shapes.
    """
    import dust3r.utils.misc as misc

    _orig_transpose = misc.transpose_to_landscape

    def patched_transpose_to_landscape(head, activate=True):
        if not activate:

            def wrapper_no(decout, true_shape):
                return head(decout, (IMG_HEIGHT, IMG_WIDTH))

            return wrapper_no
        return _orig_transpose(head, activate)

    def patched_is_symmetrized(gt1, gt2):
        return False

    misc.transpose_to_landscape = patched_transpose_to_landscape
    misc.is_symmetrized = patched_is_symmetrized

    for mod_name in list(sys.modules):
        mod = sys.modules.get(mod_name)
        if not mod:
            continue
        if getattr(mod, "transpose_to_landscape", None) is _orig_transpose:
            mod.transpose_to_landscape = patched_transpose_to_landscape
        if (
            hasattr(mod, "is_symmetrized")
            and mod is not misc
            and callable(getattr(mod, "is_symmetrized"))
        ):
            mod.is_symmetrized = patched_is_symmetrized


class MASt3RWrapper(torch.nn.Module):
    """Wrap MASt3R to use plain tensor inputs.

    Constructs the view dicts internally with fixed shapes so that
    dynamo can trace through the model without data-dependent control flow.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img1, img2):
        B = img1.shape[0]
        shape = torch.tensor([[IMG_HEIGHT, IMG_WIDTH]]).expand(B, -1)
        instance = torch.arange(B)
        view1 = {"img": img1.float(), "true_shape": shape, "instance": instance}
        view2 = {"img": img2.float(), "true_shape": shape, "instance": instance}
        return self.model(view1, view2)


class ModelVariant(StrEnum):
    """Available MASt3R model variants."""

    VIT_LARGE_BASE_DECODER_512 = "ViTLarge_BaseDecoder_512"


class ModelLoader(ForgeModel):
    """MASt3R stereo 3D reconstruction model loader implementation."""

    _VARIANTS = {
        ModelVariant.VIT_LARGE_BASE_DECODER_512: ModelConfig(
            pretrained_model_name="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_LARGE_BASE_DECODER_512

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MASt3R",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MASt3R model instance."""
        _ensure_mast3r_importable()
        _patch_dust3r_for_dynamo()
        from mast3r.model import AsymmetricMASt3R

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = AsymmetricMASt3R.from_pretrained(pretrained_model_name)
        model.eval()

        return MASt3RWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample stereo image pair inputs for the MASt3R model."""
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)

        img1 = torch.randn(batch_size, 3, IMG_HEIGHT, IMG_WIDTH, dtype=dtype)
        img2 = torch.randn(batch_size, 3, IMG_HEIGHT, IMG_WIDTH, dtype=dtype)

        return [img1, img2]
