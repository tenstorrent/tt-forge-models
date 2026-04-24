# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UniDepth V2old model loader implementation for monocular metric depth estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass
from datasets import load_dataset

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class UniDepthV2oldWrapper(nn.Module):
    """Wrapper around UniDepthV2old for depth estimation inference."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        # infer uses antialias interpolation which requires float32
        outputs = self.model.infer(image.float())
        return outputs["depth"]


@dataclass
class UniDepthV2oldConfig(ModelConfig):
    """Configuration specific to UniDepth V2old models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available UniDepth V2old model variants."""

    VIT_L14 = "ViT-L14"


class ModelLoader(ForgeModel):
    """UniDepth V2old model loader implementation."""

    _VARIANTS = {
        ModelVariant.VIT_L14: UniDepthV2oldConfig(
            pretrained_model_name="lpiccinelli/unidepth-v2old-vitl14",
            source=ModelSource.HUGGING_FACE,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_L14

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="UniDepthV2old",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import unidepth.layers.nystrom_attention as _nystrom_mod
        from unidepth.models import UniDepthV2old

        # xformers NystromAttention is unavailable on CPU; patch with standard SDPA.
        class _NystromFallback:
            def __init__(self, num_landmarks, num_heads, dropout=0.0):
                self.scale = None

            def __call__(self, q, k, v, key_padding_mask=None):
                # q, k, v: (b, n, h, d)
                b, n, h, d = q.shape
                scale = d**-0.5
                q = q.permute(0, 2, 1, 3).reshape(b * h, n, d)
                k = k.permute(0, 2, 1, 3).reshape(b * h, n, d)
                v = v.permute(0, 2, 1, 3).reshape(b * h, n, d)
                attn = torch.softmax(
                    torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1
                )
                out = torch.matmul(attn, v)
                return out.reshape(b, h, n, d).permute(0, 2, 1, 3)

        _nystrom_mod.NystromAttention = _NystromFallback

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = UniDepthV2old.from_pretrained(pretrained_model_name)
        model.eval()

        wrapper = UniDepthV2oldWrapper(model)
        wrapper.eval()

        # Keep model in float32: infer() uses antialias interpolation which
        # requires float32 and fails with bfloat16.
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        # Pass raw [0, 255] float32 values; infer() handles normalization/resizing
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0)

        if batch_size > 1:
            rgb = rgb.expand(batch_size, -1, -1, -1)

        return rgb
