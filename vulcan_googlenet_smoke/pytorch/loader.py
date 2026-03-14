# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Branch-pinned proof loader for Vulcan discovery and selection."""

from dataclasses import dataclass
from typing import Optional

import torch
from torchvision import models

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import VisionPostprocessor, VisionPreprocessor


@dataclass
class VulcanGoogleNetSmokeConfig(ModelConfig):
    source: ModelSource


class ModelVariant(StrEnum):
    GOOGLENET = "proof"


class ModelLoader(ForgeModel):
    """Small isolated proof loader for branch-pinned Vulcan validation flows."""

    _VARIANTS = {
        ModelVariant.GOOGLENET: VulcanGoogleNetSmokeConfig(
            pretrained_model_name="googlenet",
            source=ModelSource.TORCHVISION,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GOOGLENET

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
            model="VulcanGoogleNetSmoke",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=cls._VARIANTS[variant].source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        weights = models.GoogLeNet_Weights.DEFAULT
        model = models.googlenet(weights=weights)
        model.eval()
        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)
        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if self._preprocessor is None:
            self._preprocessor = VisionPreprocessor(
                model_source=self._variant_config.source,
                model_name=self._variant_config.pretrained_model_name,
                high_res_size=None,
                weight_class_name_fn=lambda _name: "GoogLeNet_Weights",
            )
            if self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=self.model,
        )

    def output_postprocess(
        self,
        output=None,
        co_out=None,
        framework_model=None,
        compiled_model=None,
        inputs=None,
        dtype_override=None,
    ):
        if self._postprocessor is None:
            self._postprocessor = VisionPostprocessor(
                model_source=self._variant_config.source,
                model_name=self._variant_config.pretrained_model_name,
                model_instance=self.model,
            )

        if output is not None:
            return self._postprocessor.postprocess(output, top_k=1, return_dict=True)

        self._postprocessor.print_results(
            co_out=co_out,
            framework_model=framework_model,
            compiled_model=compiled_model,
            inputs=inputs,
            dtype_override=dtype_override,
        )
        return None
