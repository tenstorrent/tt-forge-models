# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Large v3 GGUF model loader implementation for automatic speech recognition.

Repositories:
- https://huggingface.co/vonjack/whisper-large-v3-gguf
- https://huggingface.co/oxide-lab/whisper-large-v3-GGUF
"""
import inspect

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
)
from typing import Optional

from ...base import ForgeModel


def _fix_gguf_load_compat():
    """Fix stale GGUF patches that omit model_to_load (added in transformers 5.x).

    Some other model loaders monkey-patch load_gguf_checkpoint with a signature
    that pre-dates the model_to_load parameter. We find the real transformers
    function (identified by an explicit model_to_load parameter — not just **kwargs)
    by recursively traversing closures and module globals, then replace the module
    attribute with a direct wrapper around it.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils_mod

    def _is_real(fn):
        """True only for the genuine transformers function with explicit model_to_load."""
        try:
            return "model_to_load" in inspect.signature(fn).parameters
        except (ValueError, TypeError):
            return False

    current = gguf_utils_mod.load_gguf_checkpoint
    if _is_real(current):
        return

    def _find_real(fn, visited=None, depth=0):
        if visited is None:
            visited = set()
        fn_id = id(fn)
        if fn_id in visited or depth > 60:
            return None
        visited.add(fn_id)

        if _is_real(fn):
            return fn

        if hasattr(fn, "__closure__") and fn.__closure__:
            for cell in fn.__closure__:
                try:
                    c = cell.cell_contents
                    if callable(c):
                        result = _find_real(c, visited, depth + 1)
                        if result is not None:
                            return result
                except ValueError:
                    pass

        if hasattr(fn, "__globals__"):
            for name, val in fn.__globals__.items():
                if (
                    callable(val)
                    and ("gguf" in name.lower() or "orig" in name.lower())
                    and id(val) not in visited
                ):
                    result = _find_real(val, visited, depth + 1)
                    if result is not None:
                        return result

        return None

    real_fn = _find_real(current)
    if real_fn is None:
        return

    _real = real_fn

    def _compat(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
        return _real(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )

    gguf_utils_mod.load_gguf_checkpoint = _compat


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
        _fix_gguf_load_compat()

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
