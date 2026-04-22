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
    import site

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

    # The project has its own `evo/` model directory that shadows the pip-installed
    # evo trajectory-evaluation package required by dust3r. Pre-load the real evo
    # from site-packages so dust3r's `import evo.main_ape` resolves via sys.modules.
    site_pkgs = site.getsitepackages()
    real_evo_loaded = "evo" in sys.modules and any(
        sp in (getattr(sys.modules["evo"], "__file__", "") or "") for sp in site_pkgs
    )
    if not real_evo_loaded:
        for sp in site_pkgs:
            if os.path.isdir(os.path.join(sp, "evo")):
                for key in list(sys.modules.keys()):
                    if key == "evo" or key.startswith("evo."):
                        del sys.modules[key]
                sys.path.insert(0, sp)
                try:
                    import evo.main_ape  # noqa: F401
                finally:
                    sys.path.remove(sp)
                break

    # dust3r/model.py imports load_RAFT from third_party.raft at module level but
    # never calls it during model loading. Inject a stub to avoid the RAFT
    # dependency cascade which conflicts with the project's `utils` package.
    import types

    if "third_party.raft" not in sys.modules or not hasattr(
        sys.modules["third_party.raft"], "load_RAFT"
    ):
        raft_stub = types.ModuleType("third_party.raft")
        raft_stub.load_RAFT = None
        sys.modules["third_party.raft"] = raft_stub
        if "third_party" in sys.modules:
            sys.modules["third_party"].raft = raft_stub


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
        from dust3r.patch_embed import ManyAR_PatchEmbed

        # dust3r hardcodes .float() in ManyAR_PatchEmbed.forward which breaks
        # non-float32 dtypes. Patch it to use .to(x.dtype) instead.
        def _forward_dtype_safe(self, img, true_shape):
            B, C, H, W = img.shape
            assert W >= H, f"img should be in landscape mode, but got {W=} {H=}"
            assert H % self.patch_size[0] == 0
            assert W % self.patch_size[1] == 0
            assert true_shape.shape == (B, 2)
            W //= self.patch_size[0]
            H //= self.patch_size[1]
            n_tokens = H * W
            height, width = true_shape.T
            is_landscape = width >= height
            is_portrait = ~is_landscape
            x = img.new_zeros((B, n_tokens, self.embed_dim))
            pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)
            x[is_landscape] = (
                self.proj(img[is_landscape])
                .permute(0, 2, 3, 1)
                .flatten(1, 2)
                .to(x.dtype)
            )
            x[is_portrait] = (
                self.proj(img[is_portrait].swapaxes(-1, -2))
                .permute(0, 2, 3, 1)
                .flatten(1, 2)
                .to(x.dtype)
            )
            pos[is_landscape] = self.position_getter(1, H, W, pos.device)
            pos[is_portrait] = self.position_getter(1, W, H, pos.device)
            x = self.norm(x)
            return x, pos

        ManyAR_PatchEmbed.forward = _forward_dtype_safe

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
            "instance": [f"view1_{i}" for i in range(batch_size)],
        }
        view2 = {
            "img": torch.randn(batch_size, 3, height, width, dtype=dtype),
            "true_shape": torch.tensor([[height, width]] * batch_size),
            "instance": [f"view2_{i}" for i in range(batch_size)],
        }

        return {"view1": view1, "view2": view2}
