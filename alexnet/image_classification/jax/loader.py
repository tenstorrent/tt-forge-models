# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlexNet model loader implementation for image classification.
"""

from typing import Optional
import jax.numpy as jnp

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src import AlexNetModel, AlexNetMultichipModel


class ModelVariant(StrEnum):
    """Available AlexNet model variants."""

    CUSTOM = "custom"
    CUSTOM_1X2 = "custom_1x2"
    CUSTOM_1X4 = "custom_1x4"
    CUSTOM_1X8 = "custom_1x8"


class ModelLoader(ForgeModel):
    """AlexNet model loader implementation for image classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.CUSTOM: ModelConfig(
            pretrained_model_name="custom",
        ),
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="custom_1x2",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="custom_1x4",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="custom_1x8",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CUSTOM

    # Constants for multi-chip configuration
    ALEXNET_INPUT_SEED = 42
    ALEXNET_PARAMS_INIT_SEED = 0

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        axis_name: str = "X",
        num_devices: Optional[int] = None,
        train_mode: bool = False,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            axis_name: Name of the axis for multi-device parallelism (for multichip variants)
            num_devices: Number of devices for multi-chip execution (for multichip variants)
            train_mode: Whether model is in training mode (for multichip variants)
        """
        super().__init__(variant)
        self.axis_name = axis_name
        self.num_devices = num_devices
        self.train_mode = train_mode
        self.params_init_seed = self.ALEXNET_PARAMS_INIT_SEED

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to get info for.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="alexnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the AlexNet model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        # Check if this is a multichip variant
        is_multichip = self._variant in [
            ModelVariant.CUSTOM_1X2,
            ModelVariant.CUSTOM_1X4,
            ModelVariant.CUSTOM_1X8,
        ]

        if is_multichip:
            # Determine number of devices from variant if not explicitly set
            if self.num_devices is None:
                if self._variant == ModelVariant.CUSTOM_1X2:
                    self.num_devices = 2
                elif self._variant == ModelVariant.CUSTOM_1X4:
                    self.num_devices = 4
                elif self._variant == ModelVariant.CUSTOM_1X8:
                    self.num_devices = 8

            # Load multichip model
            if dtype_override is not None:
                model = AlexNetMultichipModel(
                    axis_name=self.axis_name,
                    num_devices=self.num_devices,
                    train_mode=self.train_mode,
                    param_dtype=dtype_override,
                )
            else:
                model = AlexNetMultichipModel(
                    axis_name=self.axis_name,
                    num_devices=self.num_devices,
                    train_mode=self.train_mode,
                )
        else:
            # Load single-chip model
            if dtype_override is not None:
                model = AlexNetModel(param_dtype=dtype_override)
            else:
                model = AlexNetModel()

        return model

    def load_inputs(self, dtype_override=None, use_random: bool = False):
        """Load and return sample inputs for the AlexNet model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
            use_random: If True, generate random inputs instead of loading from dataset.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        # Check if this is a multichip variant (which typically uses random inputs)
        is_multichip = self._variant in [
            ModelVariant.CUSTOM_1X2,
            ModelVariant.CUSTOM_1X4,
            ModelVariant.CUSTOM_1X8,
        ]

        if use_random or is_multichip:
            # Generate random inputs for testing
            import jax
            
            img = jax.random.randint(
                jax.random.PRNGKey(self.ALEXNET_INPUT_SEED),
                shape=(4, 3, 224, 224),  # NCHW format
                dtype=dtype_override if dtype_override is not None else jnp.bfloat16,
                minval=0,
                maxval=128,
            )
            return img
        else:
            # Load real image from dataset for single-chip
            from datasets import load_dataset
            import numpy as np

            # Load a sample image from open-source cats-image dataset
            dataset = load_dataset("huggingface/cats-image", split="test")
            sample = dataset[0]

            # Resize to 224x224 (AlexNet input size)
            image = sample["image"].resize((224, 224))
            image = np.array(image)

            # Normalize to [-128, 127] range as per original paper
            image = image.astype(np.float32)
            image = image - 128.0
            image = np.clip(image, -128, 127)

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            # Convert to JAX array
            inputs = jnp.array(image)

            # Apply dtype override if specified
            if dtype_override is not None:
                inputs = inputs.astype(dtype_override)

            return inputs
