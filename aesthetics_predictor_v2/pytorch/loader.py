# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Aesthetics Predictor V2 model loader implementation for image aesthetic score prediction.
"""

import torch
from transformers import AutoConfig, CLIPProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from datasets import load_dataset
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
    """Available Aesthetics Predictor V2 model variants."""

    SAC_LOGOS_AVA1_L14_LINEARMSE = "SAC_Logos_AVA1_L14_linearMSE"


class ModelLoader(ForgeModel):
    """Aesthetics Predictor V2 model loader for image aesthetic score prediction."""

    _VARIANTS = {
        ModelVariant.SAC_LOGOS_AVA1_L14_LINEARMSE: ModelConfig(
            pretrained_model_name="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAC_LOGOS_AVA1_L14_LINEARMSE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Aesthetics Predictor V2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = CLIPProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers>=5.2 enforces config_class consistency in AutoModel.register,
        # but the remote modeling_v2.py declares config_class=CLIPVisionConfig while
        # the hub config is AestheticsPredictorConfig. Load the class directly to
        # bypass the registration check.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        auto_map = getattr(config, "auto_map", {})
        model_class_ref = auto_map.get("AutoModel")
        if model_class_ref:
            model_class = get_class_from_dynamic_module(
                model_class_ref, pretrained_model_name
            )
            model = model_class.from_pretrained(
                pretrained_model_name, config=config, **model_kwargs
            )
        else:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                pretrained_model_name, trust_remote_code=True, **model_kwargs
            )

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
