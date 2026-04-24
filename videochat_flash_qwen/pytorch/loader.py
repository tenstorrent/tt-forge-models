# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoChatFlash-Qwen model loader implementation for multimodal video/image conditional generation.
"""

import contextlib
from typing import Optional

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

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
from ...tools.utils import cast_input_to_type


@contextlib.contextmanager
def _no_meta_init():
    """Patch get_init_context to skip meta device so .item() works during model init."""
    original = PreTrainedModel.get_init_context

    @classmethod
    def _patched(cls, dtype, is_quantized, _is_ds_init_called):
        contexts = original.__func__(cls, dtype, is_quantized, _is_ds_init_called)
        return [
            c
            for c in contexts
            if not (isinstance(c, torch.device) and c.type == "meta")
        ]

    PreTrainedModel.get_init_context = _patched
    try:
        yield
    finally:
        PreTrainedModel.get_init_context = original


class ModelVariant(StrEnum):
    """Available VideoChatFlash-Qwen model variants."""

    TINY = "tiny"
    QWEN2_7B_RES448 = "QWEN2_7B_RES448"


class ModelLoader(ForgeModel):
    """VideoChatFlash-Qwen model loader for multimodal video/image conditional generation."""

    _VARIANTS = {
        ModelVariant.TINY: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-videochat-flash-qwen",
        ),
        ModelVariant.QWEN2_7B_RES448: ModelConfig(
            pretrained_model_name="OpenGVLab/VideoChat-Flash-Qwen2-7B_res448",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VideoChatFlash-Qwen model loader."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoChatFlash-Qwen",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VideoChatFlash-Qwen model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(str(model_name), trust_remote_code=True)
        # Custom model code uses config.rope_theta but newer transformers dropped this attribute
        if not hasattr(config, "rope_theta"):
            config.rope_theta = 1000000.0

        # Skip meta-device init: model's vision tower calls .item() during __init__
        # which fails on meta tensors. Loading without meta device uses more memory
        # but avoids the error.
        with _no_meta_init():
            model = AutoModel.from_pretrained(
                str(model_name), config=config, **model_kwargs
            )

        model.eval()

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VideoChatFlash-Qwen."""
        if self.tokenizer is None:
            self._load_tokenizer()

        text_prompt = "<video>\nWhat is shown in this video?"
        inputs = self.tokenizer(text_prompt, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        # Provide dummy video: 4 frames at 448x448 (mm_local_num_frames=4)
        video_dtype = dtype_override if dtype_override else torch.float32
        dummy_video = torch.zeros(4, 3, 448, 448, dtype=video_dtype)
        inputs["images"] = [dummy_video]
        inputs["modalities"] = ["video"]

        return dict(inputs)
