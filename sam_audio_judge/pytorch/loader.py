# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Facebook SAM Audio Judge model loader for audio quality evaluation.
"""

import importlib
import os
import sys
from typing import Optional
from unittest.mock import MagicMock

import torch

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


def _import_sam_audio_package():
    """Import the real sam_audio pip package, bypassing local directory shadowing.

    The local tt_forge_models/sam_audio/ model directory shadows the pip-installed
    sam_audio package. This function reorders sys.path and mocks CUDA-only
    transitive dependencies (xformers, torchcodec, dacvae) so the package can
    be imported on CPU-only compile systems.
    """
    site_packages = [p for p in sys.path if "site-packages" in p]
    if not site_packages:
        raise ImportError("No site-packages found in sys.path")

    sp_base = site_packages[0]

    core_mods = set()
    core_dir = os.path.join(sp_base, "core")
    if os.path.isdir(core_dir):
        for root, _dirs, files in os.walk(core_dir):
            rel = root[len(sp_base) + 1 :].replace("/", ".")
            core_mods.add(rel)
            for f in files:
                if f.endswith(".py") and f != "__init__.py":
                    core_mods.add(f"{rel}.{f[:-3]}")

    mock_mods = list(core_mods) + [
        "xformers",
        "xformers.ops",
        "xformers.ops.fmha",
        "xformers.flash_attn_3",
        "torchcodec",
        "torchcodec.decoders",
        "torchcodec.encoders",
        "torchcodec.samplers",
        "torchcodec.transforms",
        "torchcodec._core",
        "torchcodec._core.ops",
        "torchcodec._core._metadata",
        "torchcodec._core._decoder_utils",
        "torchcodec._internally_replaced_utils",
        "dacvae",
        "audiotools",
        "imagebind",
        "imagebind.data",
        "imagebind.models",
        "imagebind.models.imagebind_model",
    ]

    for name in mock_mods:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    for k in list(sys.modules):
        if k == "sam_audio" or k.startswith("sam_audio."):
            del sys.modules[k]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    worktree_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    other = [
        p
        for p in sys.path
        if "site-packages" not in p and p != worktree_root and p != ""
    ]
    orig_path = sys.path[:]
    sys.path = site_packages + other

    try:
        return importlib.import_module("sam_audio")
    finally:
        sys.path[:] = orig_path


class ModelVariant(StrEnum):
    """Available SAM Audio Judge model variants."""

    DEFAULT = "Default"


class SAMAudioJudgeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        return output.overall


class ModelLoader(ForgeModel):
    """Facebook SAM Audio Judge model loader for audio quality evaluation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="facebook/sam-audio-judge",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SAMAudioJudge",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_AUDIO_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        sam_audio = _import_sam_audio_package()
        self._processor = sam_audio.SAMAudioJudgeProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        sam_audio = _import_sam_audio_package()
        SAMAudioJudgeModel = sam_audio.SAMAudioJudgeModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = SAMAudioJudgeModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return SAMAudioJudgeWrapper(model)

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._load_processor()

        sampling_rate = 16000
        duration_seconds = 1
        num_samples = sampling_rate * duration_seconds

        input_audio = torch.randn(1, num_samples)
        separated_audio = torch.randn(1, num_samples)
        description = "A person speaking"

        inputs = self._processor(
            text=[description],
            input_audio=[input_audio],
            separated_audio=[separated_audio],
        )

        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        return inputs
