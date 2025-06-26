# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Openpose model loader implementation
"""
import torch
from diffusers.utils import load_image
from ...base import ForgeModel

dependencies = ["controlnet_aux==0.0.9"]


class ModelLoader(ForgeModel):
    """Openpose model loader implementation."""

    # Shared configuration parameters
    model_name = "lllyasviel/ControlNet"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Openpose model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Openpose model instance.

        """
        from controlnet_aux import OpenposeDetector

        model = OpenposeDetector.from_pretrained(cls.model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, batch_size=1):
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
