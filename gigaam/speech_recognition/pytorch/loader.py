# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GigaAM-v3 model loader implementation for automatic speech recognition.

GigaAM-v3 is a Conformer-based ASR foundation model pretrained on 700k hours
of Russian speech. It supports CTC and RNN-T decoding variants.
"""

import sys
import types
from functools import wraps
from typing import Optional

import torch

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GigaAM-v3 speech recognition model variants."""

    CTC = "CTC"
    RNNT = "RNNT"


class ModelLoader(ForgeModel):
    """GigaAM-v3 model loader implementation for automatic speech recognition."""

    _VARIANTS = {
        ModelVariant.CTC: ModelConfig(
            pretrained_model_name="ai-sage/GigaAM-v3",
        ),
        ModelVariant.RNNT: ModelConfig(
            pretrained_model_name="ai-sage/GigaAM-v3",
        ),
    }

    # Map variant enum to HuggingFace revision branch
    _REVISION_MAP = {
        ModelVariant.CTC: "ctc",
        ModelVariant.RNNT: "rnnt",
    }

    DEFAULT_VARIANT = ModelVariant.CTC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GigaAM-v3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_torchaudio():
        # libtorchaudio.so links against libtorch_cuda.so which is absent in
        # CPU-only TT environments.  Stub out the C-extension gate so the pure-
        # Python torchaudio.transforms (MelSpectrogram) can be imported.
        if "torchaudio._extension" in sys.modules:
            return

        def _fail_decorator(msg):
            def decorator(func):
                @wraps(func)
                def wrapped(*args, **kwargs):
                    raise RuntimeError(msg)

                return wrapped

            return decorator

        ext = types.ModuleType("torchaudio._extension")
        ext._IS_TORCHAUDIO_EXT_AVAILABLE = False
        ext._IS_RIR_AVAILABLE = False
        ext._IS_ALIGN_AVAILABLE = False
        ext.fail_if_no_rir = _fail_decorator("RIR extension not available")
        ext.fail_if_no_align = _fail_decorator("Align extension not available")
        ext._check_cuda_version = lambda: None
        sys.modules["torchaudio._extension"] = ext

    @staticmethod
    def _patch_pyannote():
        # pyannote.audio is only used in get_vad_pipeline() for long-form
        # transcription, not in the standard inference forward path.
        # Stub the top-level package so transformers check_imports passes.
        if "pyannote" not in sys.modules:
            sys.modules["pyannote"] = types.ModuleType("pyannote")

    def load_model(self, *, dtype_override=None, **kwargs):
        self._patch_torchaudio()
        self._patch_pyannote()

        from transformers import AutoModel, PreTrainedModel

        revision = self._REVISION_MAP[self._variant]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # torchaudio.functional.melscale_fbanks calls .any() (which internally
        # calls .item()) on tensors created during FeatureExtractor.__init__.
        # Under the meta-device context used by transformers 5.x this raises
        # "Tensor.item() cannot be called on meta tensors".  Temporarily remove
        # the meta device from get_init_context so the filterbank is computed on
        # real tensors.
        #
        # GigaAMModel.__init__ was written for an older transformers API: it does
        # not call self.post_init(), which transformers 5.x requires to set up
        # all_tied_weights_keys before _finalize_model_loading runs.  Patch
        # _finalize_model_loading to call post_init() when the attribute is absent.
        _orig_get_init = PreTrainedModel.get_init_context
        _orig_finalize = PreTrainedModel._finalize_model_loading

        @classmethod  # type: ignore[misc]
        def _no_meta(cls, dtype, is_quantized, _is_ds_init_called):
            return [
                c
                for c in _orig_get_init.__func__(cls, dtype, is_quantized, _is_ds_init_called)
                if not isinstance(c, torch.device)
            ]

        @staticmethod  # type: ignore[misc]
        def _finalize_with_post_init(model, load_config, loading_info):
            if not hasattr(model, "all_tied_weights_keys"):
                model.post_init()
            return _orig_finalize(model, load_config, loading_info)

        PreTrainedModel.get_init_context = _no_meta
        PreTrainedModel._finalize_model_loading = _finalize_with_post_init
        try:
            model = AutoModel.from_pretrained(
                self._variant_config.pretrained_model_name,
                revision=revision,
                trust_remote_code=True,
                **model_kwargs,
            )
        finally:
            PreTrainedModel.get_init_context = _orig_get_init
            PreTrainedModel._finalize_model_loading = _orig_finalize

        model.eval()

        # MelScale.fb is a buffer that gets cast to bfloat16 when the model is
        # loaded with torch_dtype=bfloat16, but STFT always outputs float32.
        # Cast the preprocessor back to float32 so the mel matmul dtype matches.
        if hasattr(model, "model") and hasattr(model.model, "preprocessor"):
            model.model.preprocessor.float()

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # GigaAM-v3 expects raw audio at 16kHz sample rate
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)

        # Do not cast audio to dtype_override: the preprocessor runs STFT which
        # requires float32 (MKL FFT rejects BFloat16).  The encoder handles its
        # own precision via autocast on non-CPU devices.

        # GigaAMModel.forward() requires (features, feature_lengths)
        feature_lengths = torch.tensor([audio_tensor.shape[-1]], dtype=torch.long)

        return [audio_tensor, feature_lengths]
