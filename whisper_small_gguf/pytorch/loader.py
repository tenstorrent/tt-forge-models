# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Small GGUF model loader implementation for automatic speech recognition.

Repository:
- https://huggingface.co/FL33TW00D-HF/whisper-small
"""
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
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


class ModelVariant(StrEnum):
    """Available Whisper Small GGUF model variants."""

    Q4_K = "Q4_K"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """Whisper Small GGUF model loader implementation for automatic speech recognition tasks."""

    _VARIANTS = {
        ModelVariant.Q4_K: ModelConfig(
            pretrained_model_name="FL33TW00D-HF/whisper-small",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="FL33TW00D-HF/whisper-small",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K: "small_q4k.gguf",
        ModelVariant.Q8_0: "small_q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper Small GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from gguf import GGUFReader, dequantize

        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        config = WhisperConfig.from_pretrained("openai/whisper-small")
        config.use_cache = False
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")

        local_path = hf_hub_download(repo_id=pretrained_model_name, filename=gguf_file)
        reader = GGUFReader(local_path)

        target_dtype = dtype_override if dtype_override is not None else torch.float32
        state_dict = {}
        for tensor in reader.tensors:
            arr = dequantize(tensor.data, tensor.tensor_type)
            state_dict[tensor.name] = torch.from_numpy(arr).to(target_dtype)

        model = WhisperForConditionalGeneration(config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained("openai/whisper-small")

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic audio waveform (1 second at 16kHz)
        sampling_rate = 16000
        sample_audio = torch.randn(sampling_rate).numpy()

        processor = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=sampling_rate
        )
        input_features = processor.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return [input_features, decoder_input_ids]
