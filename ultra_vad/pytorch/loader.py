# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UltraVAD model loader implementation for context-aware audio endpointing.
"""

import numpy as np
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
    """Available UltraVAD model variants."""

    ULTRAVAD = "ultraVAD"


class ModelLoader(ForgeModel):
    """UltraVAD model loader implementation for audio-native endpointing tasks."""

    _VARIANTS = {
        ModelVariant.ULTRAVAD: ModelConfig(
            pretrained_model_name="fixie-ai/ultraVAD",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ULTRAVAD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="UltraVAD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        import transformers
        from transformers import AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        pretrained_model_name = self._variant_config.pretrained_model_name

        # transformers 5.x requires FeatureExtractionMixin for audio_processor in
        # ProcessorMixin.__init__, but UltravoxProcessor passes a WhisperProcessor
        # (a ProcessorMixin). Patch check_argument_for_proper_class to accept this
        # legacy pattern so the rest of the class continues to work as intended.
        try:
            _UltravoxProcessor = get_class_from_dynamic_module(
                "ultravox_processing.UltravoxProcessor", pretrained_model_name
            )
            if not getattr(
                _UltravoxProcessor.check_argument_for_proper_class, "_patched", False
            ):
                _orig_check = _UltravoxProcessor.check_argument_for_proper_class

                def _check_compat(self, argument_name, argument):
                    if argument_name == "audio_processor" and isinstance(
                        argument, transformers.ProcessorMixin
                    ):
                        return type(argument)
                    return _orig_check(self, argument_name, argument)

                _check_compat._patched = True
                _UltravoxProcessor.check_argument_for_proper_class = _check_compat
        except Exception:
            pass

        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UltraVAD model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UltraVAD model instance.
        """
        import transformers
        import transformers.modeling_utils
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        pretrained_model_name = self._variant_config.pretrained_model_name

        # ultraVAD remote code was written for transformers 4.x; patch removed APIs.
        # (1) _init_weights flag must be True so the language model is loaded from its own
        #     pretrained source (the UltraVAD checkpoint has no language-model weights).
        #     In old transformers this was set to False inside no_init_weights(), but
        #     transformers 5.x no longer has that context manager.
        if not hasattr(transformers.modeling_utils, "_init_weights"):
            transformers.modeling_utils._init_weights = True

        # (2) tie_weights() gained a recompute_mapping kwarg in transformers 5.x;
        #     patch the remote class to accept and forward it.
        # (3) get_init_context() always adds torch.device("meta") in transformers 5.x,
        #     but UltravoxModel.__init__ calls from_pretrained for sub-models which
        #     fails inside a meta context.  Override to skip the meta device so the
        #     model is created directly on CPU.
        try:
            _UltravoxModel = get_class_from_dynamic_module(
                "ultravox_model.UltravoxModel", pretrained_model_name
            )
            if not getattr(_UltravoxModel.tie_weights, "_patched_kwargs", False):
                _orig_tie = _UltravoxModel.tie_weights

                def _tie_weights_compat(self, **kwargs):
                    return _orig_tie(self)

                _tie_weights_compat._patched_kwargs = True
                _UltravoxModel.tie_weights = _tie_weights_compat

            if not getattr(_UltravoxModel.get_init_context, "_patched_no_meta", False):
                from transformers.modeling_utils import local_torch_dtype
                from transformers import initialization as _init

                @classmethod
                def _get_init_context_no_meta(
                    cls, dtype, is_quantized, _is_ds_init_called
                ):
                    return [
                        local_torch_dtype(dtype, cls.__name__),
                        _init.no_tie_weights(),
                    ]

                _get_init_context_no_meta._patched_no_meta = True
                _UltravoxModel.get_init_context = _get_init_context_no_meta
        except Exception:
            pass

        # (4) transformers 5.x removed layer_head_mask from WhisperEncoderLayer.forward;
        #     the remote ModifiedWhisperEncoder still passes it.  Add a shim that accepts
        #     and silently drops the argument.
        try:
            from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

            if not getattr(WhisperEncoderLayer.forward, "_patched_head_mask", False):
                _orig_whisper_fwd = WhisperEncoderLayer.forward

                def _whisper_fwd_compat(
                    self,
                    hidden_states,
                    attention_mask,
                    layer_head_mask=None,
                    output_attentions=False,
                ):
                    return _orig_whisper_fwd(
                        self,
                        hidden_states,
                        attention_mask,
                        output_attentions=output_attentions,
                    )

                _whisper_fwd_compat._patched_head_mask = True
                WhisperEncoderLayer.forward = _whisper_fwd_compat
        except Exception:
            pass

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = transformers.AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the UltraVAD model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        # Generate a synthetic 3-second audio waveform at 16kHz representing
        # the user's turn.
        sampling_rate = 16000
        duration_seconds = 3
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        # Dialog context ending with an assistant turn; UltraVAD estimates the
        # probability that the user's subsequent audio turn is complete.
        turns = [
            {"role": "assistant", "content": "Hi, how are you?"},
            {"role": "user", "content": "<|audio|>"},
        ]

        text = self.processor.tokenizer.apply_chat_template(
            turns, add_generation_prompt=False, tokenize=False
        )

        inputs = self.processor(
            text=text,
            audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
