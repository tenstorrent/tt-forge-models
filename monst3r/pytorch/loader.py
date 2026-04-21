# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MonST3R model loader implementation for dynamic scene 3D reconstruction.

MonST3R (Monocular Structure from Motion for 3D Reconstruction) extends the
DUSt3R architecture (AsymmetricCroCo3DStereo) with training on video data
containing motion, enabling 3D geometry estimation from image pairs in
dynamic scenes.

Requires the monst3r repository to be cloned at /tmp/monst3r_repo.
"""
import os
import sys

import torch
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

MONST3R_REPO_PATH = "/tmp/monst3r_repo"


def _ensure_monst3r_importable():
    """Ensure the monst3r repo is cloned and importable."""
    if not os.path.isdir(MONST3R_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--recurse-submodules",
                "https://github.com/Junyi42/monst3r.git",
                MONST3R_REPO_PATH,
            ]
        )

    if MONST3R_REPO_PATH not in sys.path:
        sys.path.insert(0, MONST3R_REPO_PATH)


class ModelVariant(StrEnum):
    """Available MonST3R model variants."""

    PO_TA_S_W_VITLARGE_BASEDECODER_512_DPT = "PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"


class ModelLoader(ForgeModel):
    """MonST3R model loader for dynamic scene 3D reconstruction."""

    _VARIANTS = {
        ModelVariant.PO_TA_S_W_VITLARGE_BASEDECODER_512_DPT: ModelConfig(
            pretrained_model_name="Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PO_TA_S_W_VITLARGE_BASEDECODER_512_DPT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MonST3R",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MonST3R AsymmetricCroCo3DStereo model.

        Returns:
            torch.nn.Module: The MonST3R dynamic scene 3D reconstruction model.
        """
        _ensure_monst3r_importable()
        from dust3r.model import AsymmetricCroCo3DStereo

        repo_id = self._variant_config.pretrained_model_name
        model = AsymmetricCroCo3DStereo.from_pretrained(repo_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample stereo image pair inputs for MonST3R.

        MonST3R's forward method expects view1 and view2 dicts, each containing
        'img' tensors and 'true_shape' metadata for the image pair.

        Returns:
            dict: Dict with 'view1' and 'view2' keys for model(**inputs) unpacking.
        """
        dtype = dtype_override or torch.float32
        height, width = 384, 512

        view1 = {
            "img": torch.randn(batch_size, 3, height, width, dtype=dtype),
            "true_shape": torch.tensor([[height, width]] * batch_size),
        }
        view2 = {
            "img": torch.randn(batch_size, 3, height, width, dtype=dtype),
            "true_shape": torch.tensor([[height, width]] * batch_size),
        }

        return {"view1": view1, "view2": view2}
