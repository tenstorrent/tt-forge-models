# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AnyPose model loader implementation
"""

from typing import Any, Optional

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
    load_anypose_pipe,
    create_transformer_inputs,
)


class ModelVariant(StrEnum):
    """Available AnyPose model variants."""

    ANYPOSE_QWEN_2511 = "AnyPose_Qwen_2511"


class ModelLoader(ForgeModel):
    """AnyPose model loader implementation.

    AnyPose is a LoRA adapter for Qwen/Qwen-Image-Edit-2511 that transfers
    poses from one image to another while preserving the character's appearance
    and background.
    """

    _VARIANTS = {
        ModelVariant.ANYPOSE_QWEN_2511: ModelConfig(
            pretrained_model_name="lilylilith/AnyPose",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANYPOSE_QWEN_2511

    prompt = (
        "Make the person in image 1 do the exact same pose of the person in image 2. "
        "Changing the style and background of the image of the person in image 1 is "
        "undesirable, so don't do it. The new pose should be pixel accurate to the pose "
        "we are trying to copy."
    )
    base_model = "Qwen/Qwen-Image-Edit-2511"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AnyPose",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the AnyPose pipeline and return the transformer sub-module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            QwenImageTransformer2DModel: The transformer component of the pipeline.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_anypose_pipe(self.base_model, pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None):
        """Load synthetic inputs for the AnyPose transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Inputs matching QwenImageTransformer2DModel.forward() signature.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        inputs = create_transformer_inputs(self.pipeline.transformer)

        if dtype_override is not None:
            inputs["hidden_states"] = inputs["hidden_states"].to(dtype_override)
            inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"].to(
                dtype_override
            )
            inputs["timestep"] = inputs["timestep"].to(dtype_override)

        return inputs

    def unpack_forward_output(self, fwd_output: Any):
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
