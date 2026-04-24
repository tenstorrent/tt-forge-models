# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Small GGUF model loader implementation for automatic speech recognition.

Repository:
- https://huggingface.co/FL33TW00D-HF/whisper-small
"""
import importlib.metadata

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

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager.

        transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
        is installed later, the mapping is stale and version detection falls back to
        gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
        """
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        config = WhisperConfig.from_pretrained("openai/whisper-small")
        config.use_cache = False
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")

        model = WhisperForConditionalGeneration(config)

        # The FL33TW00D-HF GGUF file uses PyTorch-style tensor names and omits
        # the standard `general.architecture` metadata field, so transformers'
        # from_pretrained(gguf_file=...) path fails. Load weights directly.
        from huggingface_hub import hf_hub_download
        from gguf import GGUFReader, dequantize
        import numpy as np

        gguf_path = hf_hub_download(repo_id=pretrained_model_name, filename=gguf_file)
        reader = GGUFReader(gguf_path)
        state_dict = {}
        for tensor in reader.tensors:
            arr = np.array(tensor.data)
            dq = dequantize(arr, tensor.tensor_type)
            state_dict[tensor.name] = torch.from_numpy(dq)

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

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
        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
        }
