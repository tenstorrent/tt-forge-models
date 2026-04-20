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
    EUSTLB = "eustlb"


class ModelLoader(ForgeModel):
    """Higgs Audio V2 Tokenizer model loader implementation for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="bosonai/higgs-audio-v2-tokenizer",
        ),
        ModelVariant.EUSTLB: ModelConfig(
            pretrained_model_name="eustlb/higgs-audio-v2-tokenizer",
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
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._variant == ModelVariant.EUSTLB:
            from transformers import AutoModel

            model_kwargs = {"trust_remote_code": True}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
            model.eval()
        else:
            from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
                load_higgs_audio_tokenizer,
            )

            model = load_higgs_audio_tokenizer(pretrained_model_name, device="cpu")
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
