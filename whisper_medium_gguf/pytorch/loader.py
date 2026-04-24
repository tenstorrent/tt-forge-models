# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Medium GGUF model loader implementation for automatic speech recognition.

Repository:
- https://huggingface.co/OllmOne/whisper-medium-GGUF
"""
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
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

# The GGUF repo does not include processor files, so we load from the base model.
BASE_MODEL_NAME = "openai/whisper-medium"


def _load_state_dict_from_gguf(gguf_path, dtype=None):
    """Load state dict from a non-standard GGUF file with PyTorch-style tensor names.

    This GGUF file has no architecture metadata and stores tensors with HuggingFace
    naming conventions, so the standard transformers GGUF loading path cannot be used.
    The gguf.dequantize function handles shape transposition and returns float32 arrays
    in the correct PyTorch (row-major) layout.
    """
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(gguf_path)
    state_dict = {}
    for tensor in reader.tensors:
        arr = dequantize(tensor.data, tensor.tensor_type)
        t = torch.from_numpy(np.array(arr, dtype=np.float32))
        if dtype is not None:
            t = t.to(dtype)
        state_dict[tensor.name] = t
    return state_dict


class ModelVariant(StrEnum):
    """Available Whisper Medium GGUF model variants."""

    Q4_K = "Q4_K"


class ModelLoader(ForgeModel):
    """Whisper Medium GGUF model loader implementation for automatic speech recognition tasks."""

    _VARIANTS = {
        ModelVariant.Q4_K: ModelConfig(
            pretrained_model_name="OllmOne/whisper-medium-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K

    GGUF_FILE = "model-q4k.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper Medium GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        config = WhisperConfig.from_pretrained(BASE_MODEL_NAME)
        config.use_cache = False

        self.processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME)

        gguf_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.GGUF_FILE,
        )

        state_dict = _load_state_dict_from_gguf(gguf_path, dtype=dtype_override)

        model = WhisperForConditionalGeneration(config)
        model.load_state_dict(state_dict, strict=False)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model = model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(BASE_MODEL_NAME)

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic audio waveform (1 second at 16kHz)
        sampling_rate = 16000
        sample_audio = torch.randn(sampling_rate).numpy()

        processor_output = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=sampling_rate
        )
        input_features = processor_output.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
        }
