# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WavTokenizer GGUF model loader for discrete audio codec tokenization.

Loads the GGUF-converted WavTokenizer decoder hosted at ggml-org/WavTokenizer,
which is a GGUF packaging of novateur/WavTokenizer-large-speech-75token (a
VQ-VAE style audio codec operating at 24 kHz with 75 tokens per second).

Requires the WavTokenizer repository to be cloned at /tmp/wavtokenizer_repo.
"""

import os
import sys
from typing import Optional

import torch

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

WAVTOKENIZER_REPO_PATH = "/tmp/wavtokenizer_repo"


def _ensure_wavtokenizer_importable():
    """Ensure the WavTokenizer repo is cloned and importable."""
    if not os.path.isdir(WAVTOKENIZER_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/jishengpeng/WavTokenizer.git",
                WAVTOKENIZER_REPO_PATH,
            ]
        )

    if WAVTOKENIZER_REPO_PATH not in sys.path:
        sys.path.insert(0, WAVTOKENIZER_REPO_PATH)


class ModelVariant(StrEnum):
    """Available WavTokenizer GGUF model variants."""

    LARGE_75_F16 = "Large_75_F16"
    LARGE_75_Q5_1 = "Large_75_Q5_1"


class ModelLoader(ForgeModel):
    """WavTokenizer GGUF model loader for discrete audio codec tokenization.

    Downloads a GGUF-packaged WavTokenizer decoder checkpoint from
    ggml-org/WavTokenizer and the companion config YAML from the base
    novateur/WavTokenizer repo, then loads them via the upstream WavTokenizer
    library.
    """

    _VARIANTS = {
        ModelVariant.LARGE_75_F16: ModelConfig(
            pretrained_model_name="ggml-org/WavTokenizer",
        ),
        ModelVariant.LARGE_75_Q5_1: ModelConfig(
            pretrained_model_name="ggml-org/WavTokenizer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_75_Q5_1

    # Config YAML is hosted in the base novateur/WavTokenizer repo
    _CONFIG_REPO = "novateur/WavTokenizer"
    _CONFIG_FILENAME = (
        "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    )

    # PyTorch Lightning checkpoint from the large-speech-75token repo
    _CHECKPOINT_REPO = "novateur/WavTokenizer-large-speech-75token"
    _CHECKPOINT_FILENAME = "wavtokenizer_large_speech_320_v2.ckpt"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WavTokenizer GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-packaged WavTokenizer decoder model."""
        from huggingface_hub import hf_hub_download

        _ensure_wavtokenizer_importable()
        from decoder.pretrained import WavTokenizer

        config_path = hf_hub_download(
            repo_id=self._CONFIG_REPO, filename=self._CONFIG_FILENAME
        )

        model_path = hf_hub_download(
            repo_id=self._CHECKPOINT_REPO,
            filename=self._CHECKPOINT_FILENAME,
        )

        model = WavTokenizer.from_pretrained0802(config_path, model_path)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the WavTokenizer model.

        Returns:
            dict: Dictionary with 'wav' tensor (1-second mono audio at 24kHz)
                and 'bandwidth_id' tensor for codec configuration.
        """
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        sample_rate = 24000
        wav = torch.randn(1, 1, sample_rate, dtype=dtype)

        bandwidth_id = torch.tensor([0])

        return {"audio_input": wav, "bandwidth_id": bandwidth_id}
