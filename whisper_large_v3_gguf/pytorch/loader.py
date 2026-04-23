# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Large v3 GGUF model loader implementation for automatic speech recognition.

Repositories:
- https://huggingface.co/vonjack/whisper-large-v3-gguf
- https://huggingface.co/oxide-lab/whisper-large-v3-GGUF
"""
import numpy as np
import torch
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
    """Available Whisper Large v3 GGUF model variants."""

    F16 = "F16"
    Q8_0 = "Q8_0"
    OXIDE_LAB_Q4_0 = "oxide_lab_Q4_0"
    OXIDE_LAB_Q4_1 = "oxide_lab_Q4_1"
    OXIDE_LAB_Q8_0 = "oxide_lab_Q8_0"


class ModelLoader(ForgeModel):
    """Whisper Large v3 GGUF model loader implementation for automatic speech recognition tasks."""

    _VARIANTS = {
        ModelVariant.F16: ModelConfig(
            pretrained_model_name="vonjack/whisper-large-v3-gguf",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="vonjack/whisper-large-v3-gguf",
        ),
        ModelVariant.OXIDE_LAB_Q4_0: ModelConfig(
            pretrained_model_name="oxide-lab/whisper-large-v3-GGUF",
        ),
        ModelVariant.OXIDE_LAB_Q4_1: ModelConfig(
            pretrained_model_name="oxide-lab/whisper-large-v3-GGUF",
        ),
        ModelVariant.OXIDE_LAB_Q8_0: ModelConfig(
            pretrained_model_name="oxide-lab/whisper-large-v3-GGUF",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.F16: "whisper-large-v3-f16.gguf",
        ModelVariant.Q8_0: "whisper-large-v3-q8_0.gguf",
        ModelVariant.OXIDE_LAB_Q4_0: "whisper-large-v3-q4_0.gguf",
        ModelVariant.OXIDE_LAB_Q4_1: "whisper-large-v3-q4_1.gguf",
        ModelVariant.OXIDE_LAB_Q8_0: "whisper-large-v3-q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper Large v3 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Refresh PACKAGE_DISTRIBUTION_MAPPING so that gguf, installed at
        # test-time via requirements.txt, is visible to is_gguf_available().
        import importlib.metadata as _imeta
        import transformers.utils.import_utils as _import_utils

        _import_utils.PACKAGE_DISTRIBUTION_MAPPING = _imeta.packages_distributions()

        config = WhisperConfig.from_pretrained("openai/whisper-large-v3")
        config.use_cache = False

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

        # transformers does not support the whisper GGUF architecture via
        # from_pretrained(gguf_file=...).  The oxide-lab GGUF format stores
        # tensors under their HuggingFace names directly, so we load them
        # manually using the gguf library and bypass from_pretrained entirely.
        from huggingface_hub import hf_hub_download
        from gguf import GGUFReader, dequantize

        gguf_path = hf_hub_download(pretrained_model_name, gguf_file)

        model = WhisperForConditionalGeneration(config)

        reader = GGUFReader(gguf_path)
        state_dict = {}
        for tensor in reader.tensors:
            name = tensor.name
            if name == "mel_filters":
                continue
            # oxide-lab GGUF prefixes proj_out with "model." but HF does not
            if name == "model.proj_out.weight":
                name = "proj_out.weight"
            weights = dequantize(tensor.data, tensor.tensor_type)
            state_dict[name] = torch.from_numpy(np.copy(weights))

        model.load_state_dict(state_dict, strict=True)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model = model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained("openai/whisper-large-v3")

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
