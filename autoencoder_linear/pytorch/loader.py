# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
Autoencoder Linear model loader implementation
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import Optional
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
from .src.linear_ae import LinearAE


class Variant(StrEnum):
    """Available Autoencoder Linear model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Autoencoder linear model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        Variant.BASE: ModelConfig(
            pretrained_model_name="",  # No pretrained weights needed
        )
    }

    DEFAULT_VARIANT = Variant.BASE

    def __init__(self, variant: Optional[StrEnum] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional StrEnum specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional StrEnum specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="autoencoder_linear",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Autoencoder Linear model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Autoencoder Linear model instance.
        """
        model = LinearAE()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Autoencoder Linear model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: A batch of input tensors that can be fed to the model.
        """

        # Define transform to normalize data
        transform = transforms.Compose(
            [
                transforms.Resize((1, 784)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Load sample from MNIST dataset
        dataset = load_dataset("mnist")
        sample = dataset["train"][0]["image"]
        batch_tensor = torch.stack([transform(sample)] * batch_size)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def post_processing(self, co_out, save_path):
        output_image = co_out[0].view(1, 28, 28).detach().numpy()
        os.makedirs(save_path, exist_ok=True)
        reconstructed_image_path = f"{save_path}/reconstructed_image.png"
        plt.imsave(reconstructed_image_path, np.squeeze(output_image), cmap="gray")
