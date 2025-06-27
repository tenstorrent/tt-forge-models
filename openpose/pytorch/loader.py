# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Openpose model loader implementation
"""


from diffusers.utils import load_image
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel

dependencies = ["controlnet_aux==0.0.9"]


class ModelLoader(ForgeModel):
    """Openpose model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "lllyasviel/ControlNet"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="OpenPose",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_KEYPOINT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Openpose model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Openpose model instance.

        """
        from controlnet_aux import OpenposeDetector

        model = OpenposeDetector.from_pretrained(self.model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Openpose model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        image = load_image(
            "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png"
        )
        image = [image] * batch_size
        arguments = {"input_image": image, "hand_and_face": True}
        return arguments
