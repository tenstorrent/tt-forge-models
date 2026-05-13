# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Motif-Image-6B-Preview model loader implementation for text-to-image generation.

The HF repo `Motif-Technologies/Motif-Image-6B-Preview` is NOT a Diffusers
pipeline — it ships a custom `MotifDiT` transformer with bespoke modeling code
in `models/modeling_dit.py` and a JSON config at `configs/mmdit_xlarge_hq.json`,
plus an FSDP-saved checkpoint at `checkpoints/pytorch_model_fsdp.bin`.

This loader downloads the source + checkpoint via `huggingface_hub`, stages a
patched copy of the modeling code into a writable cache dir (the upstream code
hardcodes `device="cuda"` in three places), puts that dir on `sys.path`, then
instantiates only the `MotifDiT` (the 6B transformer that is the actual
Tenstorrent target). The full `MotifImage` wrapper additionally wires in
T5-XXL, CLIP-L, CLIP-G, and a VAE — those are CPU-side preprocessing and out
of scope for the bringup test, mirroring how upstream `inference.py` filters
checkpoint keys to `dit.*` for actual generation.

"""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Motif-Image model variants."""

    MOTIF_IMAGE_6B_PREVIEW = "Motif-Image-6B-Preview"


class ModelLoader(ForgeModel):
    """Motif-Image-6B-Preview model loader implementation.

    Targets only the MotifDiT transformer (the Tenstorrent-relevant 6B model);
    text encoders (T5-XXL, CLIP-L/G) and the VAE are out of scope.
    """

    _VARIANTS = {
        ModelVariant.MOTIF_IMAGE_6B_PREVIEW: ModelConfig(
            pretrained_model_name="Motif-Technologies/Motif-Image-6B-Preview",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOTIF_IMAGE_6B_PREVIEW

    _SRC_FILES: Tuple[str, ...] = (
        "configs/configuration_mmdit.py",
        "configs/mmdit_xlarge_hq.json",
        "models/modeling_dit.py",
    )
    _CKPT_FILE = "checkpoints/pytorch_model_fsdp.bin"
    _CONFIG_FILE = "configs/mmdit_xlarge_hq.json"
    _STAGE_DIR_NAME = "tt_xla_motif_image_6B_preview_src"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._dit_config = None
        self._stage_dir: Optional[Path] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Motif-Image",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _stage_root(self) -> Path:
        return (
            Path(
                os.environ.get(
                    "TT_XLA_CACHE_DIR",
                    Path.home() / ".cache" / "tt-xla",
                )
            )
            / self._STAGE_DIR_NAME
        )

    def _materialize_sources(self) -> Path:
        """Download upstream modeling code to a writable staging dir, patch
        out hardcoded CUDA device calls, and put the dir on sys.path so
        `from configs.configuration_mmdit import MMDiTConfig` and
        `from models.modeling_dit import MotifDiT` resolve."""
        if self._stage_dir is not None:
            return self._stage_dir

        repo_id = self._variant_config.pretrained_model_name
        stage = self._stage_root()
        stage.mkdir(parents=True, exist_ok=True)

        for relpath in self._SRC_FILES:
            cached = Path(hf_hub_download(repo_id=repo_id, filename=relpath))
            dst = stage / relpath
            dst.parent.mkdir(parents=True, exist_ok=True)
            # shutil.copy resolves symlinks (HF cache snapshots are symlinks
            # into blobs/), giving us a real, owned copy we can patch.
            shutil.copy(cached, dst)

        # The upstream code hardcodes `device="cuda"` / `.cuda()` in three
        # places (one in MotifDiT.__init__, two in attention forward paths).
        # The bringup tester loads on CPU and lifts to "tt"; on a host without
        # CUDA, the construction would error before any pytest assertion.
        modeling_dit = stage / "models" / "modeling_dit.py"
        text = modeling_dit.read_text()
        patched = text
        patched = re.sub(r'device\s*=\s*"cuda"', 'device="cpu"', patched)
        patched = patched.replace(".cuda()", "")
        if patched != text:
            modeling_dit.write_text(patched)

        # Mark the package directories as namespace packages by writing empty
        # __init__.py files — keeps absolute imports cheap and explicit.
        for sub in ("configs", "models"):
            (stage / sub / "__init__.py").touch(exist_ok=True)

        sp = str(stage)
        if sp not in sys.path:
            sys.path.insert(0, sp)

        self._stage_dir = stage
        return stage

    def _load_config(self):
        if self._dit_config is not None:
            return self._dit_config
        stage = self._materialize_sources()
        from configs.configuration_mmdit import MMDiTConfig  # noqa: E402

        self._dit_config = MMDiTConfig.from_json_file(str(stage / self._CONFIG_FILE))
        return self._dit_config

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        load_weights: bool = True,
        **kwargs,
    ):
        """Build MotifDiT from the upstream config and load DiT weights from
        the FSDP checkpoint.

        `load_weights=False` skips the ~12GB checkpoint download/load and
        returns a randomly-initialized MotifDiT — useful when the bringup
        test only needs a forward-pass smoke check on the architecture.
        """
        self._materialize_sources()
        config = self._load_config()
        from models.modeling_dit import MotifDiT  # noqa: E402

        model = MotifDiT(config)

        if load_weights:
            ckpt_path = hf_hub_download(
                repo_id=self._variant_config.pretrained_model_name,
                filename=self._CKPT_FILE,
            )
            # Upstream `inference.py` filters checkpoint keys to those
            # containing "dit". The full state dict additionally holds
            # text-encoder + VAE weights we don't load.
            state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            dit_prefix = "dit."
            dit_state_dict = {
                k[len(dit_prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(dit_prefix)
            }
            # strict=False because the FSDP checkpoint may include shadow
            # params (EMA) or buffers that don't 1:1 match MotifDiT keys.
            model.load_state_dict(dit_state_dict, strict=False)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None):
        """Synthetic inputs matching `MotifDiT.forward(latent, t, text_embs, pooled_text_embs)`.

        Shapes follow the upstream `mmdit_xlarge_hq.json` defaults
        (height = width = 1024, vae_compression = 8 → 128x128 latent grid).
        Latent channels are 16 (SD3-style; hardcoded to 16 in
        `MotifDiT.__init__: self.latent_chennels = 16`). Text embeddings
        match the SD3 convention: T5-XXL (4096-dim), CLIP-L (768-dim),
        CLIP-G (1280-dim), with a 2048-dim pooled embedding.
        """
        config = self._load_config()
        h_latent = config.height // config.vae_compression
        w_latent = config.width // config.vae_compression
        latent_channels = 16  # MotifDiT.__init__: self.latent_chennels = 16
        batch = 1
        dtype = dtype_override or torch.bfloat16

        latent = torch.randn(batch, latent_channels, h_latent, w_latent, dtype=dtype)
        # Discrete timestep — TextTimeEmbedding expects an integer-typed scalar/vector.
        t = torch.tensor([100], dtype=torch.long)
        # text_embs is a List[Tensor] consumed by TextConditionModule.forward(
        #   t5_xxl, clip_a, clip_b). Last dim of t5_xxl must be ENCODED_TEXT_DIM=4096;
        # clip_a + clip_b channels are concat-then-padded to 4096.
        t5_seq, clip_seq = 256, 77
        text_embs: List[torch.Tensor] = [
            torch.randn(batch, t5_seq, 4096, dtype=dtype),  # T5-XXL
            torch.randn(batch, clip_seq, 768, dtype=dtype),  # CLIP-L
            torch.randn(batch, clip_seq, 1280, dtype=dtype),  # CLIP-G (OpenCLIP-G)
        ]
        pooled_text_embs = torch.randn(batch, config.pooled_text_dim, dtype=dtype)
        return latent, t, text_embs, pooled_text_embs
