# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
from .src.model_utils import (
    create_dummy_inputs,
    load_controlnet_model,
)


class ModelVariant(StrEnum):
    CONTROLNET_UNION_2_1_8STEPS = "ControlNet_Union_2.1_8steps"
    CONTROLNET_UNION_2_1_LITE_2602_8STEPS = "ControlNet_Union_2.1_Lite_2602_8steps"


_VARIANT_FILENAMES = {
    ModelVariant.CONTROLNET_UNION_2_1_8STEPS: "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
    ModelVariant.CONTROLNET_UNION_2_1_LITE_2602_8STEPS: "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors",
}

_VARIANT_CONTROL_LAYERS = {
    ModelVariant.CONTROLNET_UNION_2_1_8STEPS: 15,
    ModelVariant.CONTROLNET_UNION_2_1_LITE_2602_8STEPS: 3,
}


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.CONTROLNET_UNION_2_1_8STEPS: ModelConfig(
            pretrained_model_name="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        ),
        ModelVariant.CONTROLNET_UNION_2_1_LITE_2602_8STEPS: ModelConfig(
            pretrained_model_name="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_UNION_2_1_8STEPS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z-Image-Turbo-Fun-Controlnet-Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        filename = _VARIANT_FILENAMES[self._variant]
        n_control_layers = _VARIANT_CONTROL_LAYERS[self._variant]

        self._model = load_controlnet_model(
            filename,
            n_control_layers=n_control_layers,
            n_refiner_layers=2,
        )

        if dtype_override is not None:
            self._model = self._model.to(dtype_override)

        return self._model

    def load_inputs(self, dtype_override=None):
        inputs = create_dummy_inputs()

        if dtype_override is not None:
            inputs = {k: v.to(dtype_override) for k, v in inputs.items()}

        return inputs
