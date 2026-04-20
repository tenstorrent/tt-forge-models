# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WavTokenizer GGUF model loader for discrete audio codec tokenization.

Loads quantized GGUF conversions of the WavTokenizer decoder model from
ggml-org/WavTokenizer. The GGUF weights are parsed with the `gguf` package,
dequantized to float32, and loaded into the PyTorch WavTokenizer architecture
defined by the upstream jishengpeng/WavTokenizer repository.

Repository: https://huggingface.co/ggml-org/WavTokenizer
"""

import os
import sys
import tempfile

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


def _gguf_to_state_dict(gguf_path: str) -> dict:
    """Parse a GGUF file into a torch state_dict of float32 tensors."""
    import gguf
    import numpy as np

    reader = gguf.GGUFReader(gguf_path)
    state_dict = {}
    for tensor in reader.tensors:
        data = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        array = np.asarray(data, dtype=np.float32)
        state_dict[tensor.name] = torch.from_numpy(array.copy())
    return state_dict


class ModelVariant(StrEnum):
    """Available WavTokenizer GGUF quantization variants."""

    F16 = "F16"
    Q5_1 = "Q5_1"


class ModelLoader(ForgeModel):
    """WavTokenizer GGUF model loader for discrete audio codec tokenization.

    Loads the WavTokenizer decoder architecture from the upstream PyTorch
    repository and initialises its weights from a GGUF checkpoint hosted on
    ggml-org/WavTokenizer.
    """

    _VARIANTS = {
        ModelVariant.F16: ModelConfig(
            pretrained_model_name="ggml-org/WavTokenizer",
        ),
        ModelVariant.Q5_1: ModelConfig(
            pretrained_model_name="ggml-org/WavTokenizer",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.F16: "WavTokenizer-Large-75-F16.gguf",
        ModelVariant.Q5_1: "WavTokenizer-Large-75-Q5_1.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.F16

    # Config YAML and checkpoint template live in the base novateur/WavTokenizer repo
    _CONFIG_REPO = "novateur/WavTokenizer"
    _CONFIG_FILENAME = (
        "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    )

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
        """Load and return the WavTokenizer decoder initialised from GGUF weights."""
        from huggingface_hub import hf_hub_download

        _ensure_wavtokenizer_importable()
        from decoder.pretrained import WavTokenizer

        config_path = hf_hub_download(
            repo_id=self._CONFIG_REPO, filename=self._CONFIG_FILENAME
        )

        gguf_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self._GGUF_FILES[self._variant],
        )

        # Convert GGUF tensors into a torch checkpoint that from_pretrained0802
        # can consume, then load the WavTokenizer model from it.
        state_dict = _gguf_to_state_dict(gguf_path)
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            torch.save({"state_dict": state_dict}, tmp.name)
            ckpt_path = tmp.name

        try:
            model = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
        finally:
            os.unlink(ckpt_path)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the WavTokenizer GGUF model.

        Returns:
            dict: Dictionary with 'wav' tensor (1-second mono audio at 24kHz)
                and 'bandwidth_id' tensor for codec configuration.
        """
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        sample_rate = 24000
        wav = torch.randn(1, 1, sample_rate, dtype=dtype)

        bandwidth_id = torch.tensor([0])

        return {"wav": wav, "bandwidth_id": bandwidth_id}
