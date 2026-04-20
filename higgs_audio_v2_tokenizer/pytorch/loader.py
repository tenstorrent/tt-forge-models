# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Higgs Audio V2 Tokenizer model loader implementation for audio tokenization tasks
"""
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


class ModelVariant(StrEnum):
    """Available Higgs Audio V2 Tokenizer model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """Higgs Audio V2 Tokenizer model loader implementation for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="bosonai/higgs-audio-v2-tokenizer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="HiggsAudioV2Tokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _fix_sys_path():
        import sys
        from pathlib import Path

        # Remove tt_forge_models root from sys.path to avoid
        # the local 'dac/' model directory shadowing the installed dac package.
        forge_models_root = str(Path(__file__).resolve().parents[2])
        removed = []
        for p in list(sys.path):
            if p == forge_models_root:
                sys.path.remove(p)
                removed.append(p)
        return removed

    @staticmethod
    def _restore_sys_path(removed):
        import sys

        for p in removed:
            sys.path.insert(0, p)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Higgs Audio V2 Tokenizer model."""
        import inspect
        import json
        import os

        removed = self._fix_sys_path()
        try:
            from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
                HiggsAudioTokenizer,
            )
            from huggingface_hub import snapshot_download
        finally:
            self._restore_sys_path(removed)

        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer_path = snapshot_download(pretrained_model_name)
        config = json.load(open(os.path.join(tokenizer_path, "config.json")))

        acoustic = config.get("acoustic_model_config", {})
        init_kwargs = {
            "n_filters": acoustic.get("encoder_hidden_size", 64),
            "D": acoustic.get("hidden_size", 256),
            "target_bandwidths": config.get("target_bandwidths", [0.5, 1, 1.5, 2]),
            "ratios": acoustic.get("downsampling_ratios", [8, 5, 4, 2, 3]),
            "sample_rate": config.get("sample_rate", 24000),
            "bins": acoustic.get("codebook_size", 1024),
            "n_q": acoustic.get("n_codebooks", 9),
            "codebook_dim": config.get("codebook_dim", 64),
            "semantic_sample_rate": config.get("semantic_sample_rate", 16000),
            "device": "cpu",
        }

        removed = self._fix_sys_path()
        try:
            model = HiggsAudioTokenizer(**init_kwargs)
        finally:
            self._restore_sys_path(removed)

        model_path = os.path.join(tokenizer_path, "model.pth")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=False)

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Higgs Audio V2 Tokenizer model."""
        sample_rate = 24000
        duration_seconds = 1
        x = torch.randn(1, 1, sample_rate * duration_seconds)

        if dtype_override is not None:
            x = x.to(dtype=dtype_override)

        bw = 2
        return {"x": x, "bw": bw}
