# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fun-ASR-Nano model loader implementation for speech recognition (ASR).

Fun-ASR-Nano-2512 is an end-to-end speech recognition model from Tongyi Lab's
FunAudioLLM initiative, combining a SenseVoice audio encoder with a Qwen3
decoder. It supports 31 languages and achieves strong performance on
far-field and noisy speech recognition tasks.
"""

import importlib.util
import os
from typing import Optional

import numpy as np
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
    """Available Fun-ASR-Nano speech recognition model variants."""

    V2512_VLLM = "2512_vllm"


class FunASRNanoWrapper(torch.nn.Module):
    """Wrapper around FunASRNano that exposes the audio encoder forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model  # FunASRNano instance (not AutoModel)

    def forward(self, speech, speech_lengths):
        encoder_out, _ = self.model.forward_export(speech, speech_lengths)
        return encoder_out


class ModelLoader(ForgeModel):
    """Fun-ASR-Nano model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.V2512_VLLM: ModelConfig(
            pretrained_model_name="allendou/Fun-ASR-Nano-2512-vllm",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2512_VLLM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._funasr_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Fun_ASR_Nano",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_funasr_model(self, dtype_override=None):
        """Load the Fun-ASR model using the funasr package."""
        import sys

        from funasr import AutoModel

        # FunASRNano registers in funasr's model tables only when its module
        # is executed with the fun_asr_nano directory on sys.path (needed for
        # the bare "from ctc import CTC" inside that module).  funasr's
        # trust_remote_code path looks for model.py in the ModelScope
        # download dir, which ships none, so we pre-register here via
        # spec_from_file_location to avoid loading the wrong "model" package
        # that other loaders may have put on sys.path (e.g. torchxrayvision).
        _spec = importlib.util.find_spec("funasr.models.fun_asr_nano")
        if _spec and _spec.submodule_search_locations:
            _pkg_dir = list(_spec.submodule_search_locations)[0]
            _model_file = os.path.join(_pkg_dir, "model.py")
            if _pkg_dir not in sys.path:
                sys.path.append(_pkg_dir)
            _fspec = importlib.util.spec_from_file_location(
                "funasr_fun_asr_nano_model", _model_file
            )
            if _fspec is not None and "FunASRNano" not in __import__(
                "funasr.register", fromlist=["tables"]
            ).tables.model_classes:
                import types as _types

                _m = _types.ModuleType("funasr_fun_asr_nano_model")
                sys.modules["funasr_fun_asr_nano_model"] = _m
                try:
                    _fspec.loader.exec_module(_m)
                except Exception as _e:
                    print(f"[fun_asr_nano] FunASRNano pre-registration failed: {_e}")

        model_kwargs = {
            "model": self._variant_config.pretrained_model_name,
            "trust_remote_code": True,
            "device": "cpu",
            # The model config ships init_param_path as "Qwen3-0.6B" (missing
            # the Qwen/ org prefix), which transformers cannot resolve.
            "llm_conf": {"init_param_path": "Qwen/Qwen3-0.6B"},
        }

        self._funasr_model = AutoModel(**model_kwargs)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Fun-ASR-Nano model instance."""
        if self._funasr_model is None:
            self._load_funasr_model(dtype_override=dtype_override)

        model = FunASRNanoWrapper(self._funasr_model.model)
        model.eval()

        # Homogenize all parameters to float32; SenseVoice audio encoder may
        # load with mixed float32/bfloat16 weights causing Conv1d type errors.
        model.to(torch.float32)
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Fun-ASR-Nano model."""
        from funasr.utils.load_utils import extract_fbank

        if self._funasr_model is None:
            self._load_funasr_model()

        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        frontend = self._funasr_model.kwargs.get("frontend")
        speech, speech_lengths = extract_fbank(audio_array, frontend=frontend)

        if dtype_override is not None:
            speech = speech.to(dtype_override)

        return (speech, speech_lengths)
