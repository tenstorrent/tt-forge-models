# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Large v3 GGUF model loader implementation for automatic speech recognition.

Repositories:
- https://huggingface.co/vonjack/whisper-large-v3-gguf
- https://huggingface.co/oxide-lab/whisper-large-v3-GGUF
"""
import inspect as _inspect

import transformers.modeling_gguf_pytorch_utils as _gguf_utils

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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = gguf_file

        # Pre-load config from the base Whisper model so that random-weights
        # mode does not need to download/parse the GGUF file just for config.
        config = WhisperConfig.from_pretrained("openai/whisper-large-v3")
        model_kwargs["config"] = config

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

        # Refresh transformers' PACKAGE_DISTRIBUTION_MAPPING so that gguf,
        # which is installed at test-time via requirements.txt, is visible to
        # is_gguf_available().  The mapping is built at module-import time;
        # if transformers was imported before gguf was installed the entry is
        # missing, causing version.parse("N/A") to crash.
        import importlib.metadata as _imeta
        import transformers.utils.import_utils as _import_utils

        _import_utils.PACKAGE_DISTRIBUTION_MAPPING = _imeta.packages_distributions()

        # Some GGUF loaders patch load_gguf_checkpoint without model_to_load
        # support; transformers >= 5.0 passes that kwarg. Fix at call time.
        _fn = _gguf_utils.load_gguf_checkpoint
        try:
            _params = _inspect.signature(_fn).parameters
            _has_var_kw = any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in _params.values()
            )
            if "model_to_load" not in _params and not _has_var_kw:
                _orig = _fn

                def _compat(gguf_path, return_tensors=False, model_to_load=None, **kw):
                    return _orig(gguf_path, return_tensors=return_tensors)

                _gguf_utils.load_gguf_checkpoint = _compat
        except (ValueError, TypeError):
            pass

        model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        ).eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

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
