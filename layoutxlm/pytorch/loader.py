# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LayoutXLM document understanding model loader implementation (PyTorch).
"""

import torch
import torch.nn as nn
from transformers import LayoutXLMTokenizer
import transformers.models.layoutlmv2.modeling_layoutlmv2 as _layoutlmv2_module
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


class _SimpleVisualBackbone(nn.Module):
    """Lightweight replacement for detectron2-based LayoutLMv2VisualBackbone."""

    def __init__(self, config):
        super().__init__()
        pool_shape = config.image_feature_pool_shape
        out_channels = pool_shape[2] if len(pool_shape) == 3 else 256
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(pool_shape[:2])
        pixel_mean = [103.53, 116.28, 123.675]
        pixel_std = [57.375, 57.12, 58.395]
        self.register_buffer(
            "pixel_mean",
            torch.tensor(pixel_mean).view(3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(pixel_std).view(3, 1, 1),
            persistent=False,
        )
        _Ns = types.SimpleNamespace
        self.cfg = _Ns(MODEL=_Ns(PIXEL_MEAN=pixel_mean, PIXEL_STD=pixel_std))

    def forward(self, images):
        x = (images - self.pixel_mean) / self.pixel_std
        x = self.backbone(x)
        x = self.pool(x).flatten(start_dim=2).transpose(1, 2).contiguous()
        return x


_orig_requires_backends = _layoutlmv2_module.requires_backends


def _skip_detectron2(obj, backends):
    backends = [backends] if isinstance(backends, str) else list(backends)
    filtered = [b for b in backends if b != "detectron2"]
    if filtered:
        _orig_requires_backends(obj, filtered)


_layoutlmv2_module.requires_backends = _skip_detectron2
_layoutlmv2_module.LayoutLMv2VisualBackbone = _SimpleVisualBackbone

import types

_mock_detectron2 = types.ModuleType("detectron2")
_mock_layers = types.ModuleType("detectron2.layers")


class _DummyFrozenBatchNorm2d:
    pass


_mock_layers.FrozenBatchNorm2d = _DummyFrozenBatchNorm2d
_mock_layers.batch_norm = types.ModuleType("detectron2.layers.batch_norm")
_mock_layers.batch_norm.FrozenBatchNorm2d = _DummyFrozenBatchNorm2d
_mock_detectron2.layers = _mock_layers
_layoutlmv2_module.detectron2 = _mock_detectron2

from transformers import LayoutLMv2Model


class ModelVariant(StrEnum):
    """Available LayoutXLM model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """LayoutXLM document understanding model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="microsoft/layoutxlm-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LayoutXLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = LayoutXLMTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LayoutLMv2Model.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        words = ["Invoice", "Number:", "12345", "Date:", "2024-01-15"]
        boxes = [
            [100, 50, 200, 80],
            [210, 50, 330, 80],
            [340, 50, 420, 80],
            [100, 100, 180, 130],
            [190, 100, 340, 130],
        ]

        encoding = self.tokenizer(
            words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        bbox = encoding["bbox"]

        image = torch.zeros(1, 3, 224, 224)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "image": image,
        }

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "last_hidden_state"):
            return fwd_output.last_hidden_state
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output
