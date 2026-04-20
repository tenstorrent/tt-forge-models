# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
humair025/devo ONNX model loader implementation.

The Devo repository packages raw ONNX artifacts used in a speech-token TTS
pipeline: a speech-token → mel-spectrogram decoder (available in fp16 and
int8 quantizations) and a HiFT vocoder that converts mel-spectrograms to
audio waveforms.
"""

from typing import Optional

import onnx
import torch
from huggingface_hub import hf_hub_download

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

_REPO_ID = "humair025/devo"


class ModelVariant(StrEnum):
    """Available Devo ONNX model variants."""

    DECODER_FP16 = "decoder-fp16"
    DECODER_INT8 = "decoder-int8"
    HIFT = "hift"


class ModelLoader(ForgeModel):
    """humair025/devo ONNX model loader."""

    _VARIANTS = {
        ModelVariant.DECODER_FP16: ModelConfig(pretrained_model_name=_REPO_ID),
        ModelVariant.DECODER_INT8: ModelConfig(pretrained_model_name=_REPO_ID),
        ModelVariant.HIFT: ModelConfig(pretrained_model_name=_REPO_ID),
    }

    _ONNX_FILES = {
        ModelVariant.DECODER_FP16: "decoder_fp16.onnx",
        ModelVariant.DECODER_INT8: "decoder_int8.onnx",
        ModelVariant.HIFT: "hift.onnx",
    }

    DEFAULT_VARIANT = ModelVariant.DECODER_FP16

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Devo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the selected Devo ONNX artifact."""
        onnx_file = self._ONNX_FILES[self._variant]
        onnx_path = hf_hub_download(repo_id=_REPO_ID, filename=onnx_file)
        return onnx.load(onnx_path)

    def load_inputs(self, **kwargs):
        """Return synthetic sample inputs matching the selected variant's graph."""
        if self._variant == ModelVariant.HIFT:
            # HiFT vocoder: mel spectrogram (1, T, 80) -> waveform
            speech_feat = torch.randn(1, 100, 80, dtype=torch.float32)
            return {"speech_feat": speech_feat}

        # Decoder variants: token indices + global embedding -> mel spectrogram
        seq_len = 50
        embedding_dim = 128
        token_indices = torch.randint(0, 4096, (seq_len,), dtype=torch.int64)

        if self._variant == ModelVariant.DECODER_FP16:
            embedding_dtype = torch.float16
        else:
            embedding_dtype = torch.float32

        global_embedding = torch.randn(embedding_dim, dtype=embedding_dtype)
        return {
            "token_indices": token_indices,
            "global_embedding": global_embedding,
        }
