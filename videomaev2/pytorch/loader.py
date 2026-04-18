# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoMAEv2 model loader implementation for video classification.
"""

from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

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
    """Available VideoMAEv2 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """VideoMAEv2 model loader for video feature extraction."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="OpenGVLab/VideoMAEv2-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VideoMAEv2 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoMAEv2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VideoMAEv2 model instance."""
        model_name = self._variant_config.pretrained_model_name
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            str(model_name), config=config, trust_remote_code=True, **kwargs
        )
        model.eval()

        # The upstream modeling file calls F.linear with keyword-only args,
        # which breaks TTPJRT's torch_function_override that indexes into args
        # positionally. Swap F in the dynamic module with a proxy that forces
        # positional calls for linear.
        self._patch_dynamic_module_linear(type(model).__module__)

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        return model

    @staticmethod
    def _patch_dynamic_module_linear(model_module_name: str) -> None:
        import sys
        import torch.nn.functional as F_mod

        class _FProxy:
            def __getattr__(self, name):
                return getattr(F_mod, name)

            @staticmethod
            def linear(input=None, weight=None, bias=None):
                return F_mod.linear(input, weight, bias)

        root = model_module_name.rsplit(".", 1)[0]
        for name, mod in list(sys.modules.items()):
            if (name == root or name.startswith(root + ".")) and getattr(
                mod, "F", None
            ) is F_mod:
                mod.F = _FProxy()

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VideoMAEv2.

        Generates a synthetic video of 16 frames at 224x224 resolution.
        The processor outputs (B, T, C, H, W) which is permuted to (B, C, T, H, W)
        as required by the model.
        """
        if self.processor is None:
            model_name = self._variant_config.pretrained_model_name
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        # Create a synthetic video (16 frames of 224x224 RGB)
        video = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)
        ]

        inputs = self.processor(video, return_tensors="pt")

        # Permute from (B, T, C, H, W) to (B, C, T, H, W) as expected by the model
        inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4)

        if dtype_override:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        if batch_size > 1:
            inputs = {
                k: v.repeat(batch_size, *([1] * (v.dim() - 1)))
                if isinstance(v, torch.Tensor)
                else v
                for k, v in inputs.items()
            }

        return dict(inputs)
