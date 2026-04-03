# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ShuffleNetV2 model loader — torchvision image classification.
https://arxiv.org/abs/1807.11164
"""

from typing import Optional
from dataclasses import dataclass
import torch
import torchvision.models as models

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
from ...tools.utils import VisionPreprocessor, VisionPostprocessor


@dataclass
class ShuffleNetV2Config(ModelConfig):
    """Configuration specific to ShuffleNetV2 models."""

    model_fn: str  # torchvision function name, e.g. "shufflenet_v2_x1_0"
    weights_cls: str  # torchvision Weights class name, e.g. "ShuffleNet_V2_X1_0_Weights"


class ModelVariant(StrEnum):
    """Available ShuffleNetV2 model variants (output width multipliers)."""

    X0_5 = "x0_5"
    X1_0 = "x1_0"
    X1_5 = "x1_5"
    X2_0 = "x2_0"


class ModelLoader(ForgeModel):
    """ShuffleNetV2 model loader — all variants from torchvision."""

    _VARIANTS = {
        ModelVariant.X0_5: ShuffleNetV2Config(
            pretrained_model_name="shufflenet_v2_x0_5",
            model_fn="shufflenet_v2_x0_5",
            weights_cls="ShuffleNet_V2_X0_5_Weights",
        ),
        ModelVariant.X1_0: ShuffleNetV2Config(
            pretrained_model_name="shufflenet_v2_x1_0",
            model_fn="shufflenet_v2_x1_0",
            weights_cls="ShuffleNet_V2_X1_0_Weights",
        ),
        ModelVariant.X1_5: ShuffleNetV2Config(
            pretrained_model_name="shufflenet_v2_x1_5",
            model_fn="shufflenet_v2_x1_5",
            weights_cls="ShuffleNet_V2_X1_5_Weights",
        ),
        ModelVariant.X2_0: ShuffleNetV2Config(
            pretrained_model_name="shufflenet_v2_x2_0",
            model_fn="shufflenet_v2_x2_0",
            weights_cls="ShuffleNet_V2_X2_0_Weights",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.X1_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ShuffleNetV2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        cfg = self._variant_config
        weights = getattr(models, cfg.weights_cls).DEFAULT
        model_fn = getattr(models, cfg.model_fn)
        model = model_fn(weights=weights)
        model.eval()

        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)
        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def _get_preprocessor(self):
        if self._preprocessor is None:
            cfg = self._variant_config

            def weight_class_name_fn(name: str) -> str:
                # "shufflenet_v2_x1_0" -> "ShuffleNet_V2_X1_0_Weights"
                return cfg.weights_cls

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.TORCHVISION,
                model_name=cfg.pretrained_model_name,
                high_res_size=None,
                weight_class_name_fn=weight_class_name_fn,
            )
            if self.model is not None:
                self._preprocessor.set_cached_model(self.model)
        return self._preprocessor

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        preprocessor = self._get_preprocessor()
        return preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            from datasets import load_dataset

            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def output_postprocess(self, output=None, co_out=None, **kwargs):
        if self._postprocessor is None:
            cfg = self._variant_config
            self._postprocessor = VisionPostprocessor(
                model_source=ModelSource.TORCHVISION,
                model_name=cfg.pretrained_model_name,
                model_instance=self.model,
            )
        if output is not None:
            return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
        if co_out is not None:
            self._postprocessor.print_results(co_out=co_out, **kwargs)
        return None
