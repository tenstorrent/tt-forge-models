# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GigaAM v2 ONNX model loader implementation for automatic speech recognition.

GigaAM v2 is a Conformer-based Russian ASR model published by SaluteDevices.
This loader wraps the ONNX exports hosted at ``istupakov/gigaam-v2-onnx``,
which contains a single-file CTC model and the three components of an
RNN-T model (encoder, decoder, joint).
"""

from typing import Optional

import onnx
import torch

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
    """Available GigaAM v2 ONNX model variants."""

    CTC = "CTC"
    RNNT_ENCODER = "RNNT_ENCODER"
    RNNT_DECODER = "RNNT_DECODER"
    RNNT_JOINT = "RNNT_JOINT"


# Maps each variant to the ONNX file hosted in the HuggingFace repo.
_FILENAME_MAP = {
    ModelVariant.CTC: "v2_ctc.onnx",
    ModelVariant.RNNT_ENCODER: "v2_rnnt_encoder.onnx",
    ModelVariant.RNNT_DECODER: "v2_rnnt_decoder.onnx",
    ModelVariant.RNNT_JOINT: "v2_rnnt_joint.onnx",
}

# Log-mel feature dimension expected by the GigaAM v2 encoder.
_FEATURE_DIM = 64
# RNN-T decoder LSTM hidden size.
_RNNT_HIDDEN_DIM = 320
# Encoder output dimension used by the joint network.
_ENCODER_OUT_DIM = 768


class ModelLoader(ForgeModel):
    """GigaAM v2 ONNX model loader implementation for automatic speech recognition."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name="istupakov/gigaam-v2-onnx")
        for variant in ModelVariant
    }

    DEFAULT_VARIANT = ModelVariant.CTC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GigaAM-v2-ONNX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download the ONNX file for the selected variant and return the model proto."""
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=_FILENAME_MAP[self._variant],
        )
        return onnx.load(local_path)

    def load_inputs(self, **kwargs):
        """Build synthetic inputs matching the selected variant's ONNX signature."""
        # Synthetic mel-spectrogram length. ~1 second of audio at 10ms hop.
        num_frames = 100

        if self._variant == ModelVariant.CTC:
            features = torch.randn(1, _FEATURE_DIM, num_frames, dtype=torch.float32)
            feature_lengths = torch.tensor([num_frames], dtype=torch.int64)
            return {"features": features, "feature_lengths": feature_lengths}

        if self._variant == ModelVariant.RNNT_ENCODER:
            audio_signal = torch.randn(1, _FEATURE_DIM, num_frames, dtype=torch.float32)
            length = torch.tensor([num_frames], dtype=torch.int64)
            return {"audio_signal": audio_signal, "length": length}

        if self._variant == ModelVariant.RNNT_DECODER:
            x = torch.zeros(1, 1, dtype=torch.int64)
            h = torch.zeros(1, 1, _RNNT_HIDDEN_DIM, dtype=torch.float32)
            c = torch.zeros(1, 1, _RNNT_HIDDEN_DIM, dtype=torch.float32)
            return {"x": x, "h.1": h, "c.1": c}

        if self._variant == ModelVariant.RNNT_JOINT:
            encoder_outputs = torch.randn(1, _ENCODER_OUT_DIM, 1, dtype=torch.float32)
            decoder_outputs = torch.randn(1, _RNNT_HIDDEN_DIM, 1, dtype=torch.float32)
            return {
                "encoder_outputs": encoder_outputs,
                "decoder_outputs": decoder_outputs,
            }

        raise ValueError(f"Unsupported variant: {self._variant}")
