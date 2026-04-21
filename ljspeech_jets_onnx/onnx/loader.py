# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LJSpeech JETS ONNX model loader implementation for text-to-speech tasks.
"""

from typing import Optional

import onnx
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


class ModelVariant(StrEnum):
    """Available LJSpeech JETS ONNX model variants."""

    LJSPEECH_JETS_ONNX = "ljspeech-jets-onnx"


class ModelLoader(ForgeModel):
    """LJSpeech JETS ONNX model loader for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.LJSPEECH_JETS_ONNX: ModelConfig(
            pretrained_model_name="NeuML/ljspeech-jets-onnx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LJSPEECH_JETS_ONNX

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LJSpeech JETS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the LJSpeech JETS ONNX model from Hugging Face.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.onnx",
        )
        model = onnx.load(local_path)

        return model

    def load_inputs(self, **kwargs):
        """Generate sample token-ID inputs for the LJSpeech JETS ONNX model.

        The model expects a 1D int64 tensor named "text" containing phoneme
        token IDs produced by a TTS tokenizer.

        Returns:
            torch.Tensor: Sample token IDs of shape [seq_len] (int64).
        """
        seq_len = 32
        # The LJSpeech JETS token vocabulary is small (~70 phonemes); stay
        # within a conservative upper bound.
        text = torch.randint(1, 70, (seq_len,), dtype=torch.int64)
        return text
