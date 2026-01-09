# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CLIP model loader implementation for JAX.
"""

from typing import Optional
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
from ....tools.jax_utils import cast_hf_model_to_type
import flax.nnx as nnx
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import numpy as np


class ModelVariant(StrEnum):
    """Available CLIP model variants."""

    BASE_PATCH16 = "base_patch16"
    BASE_PATCH32 = "base_patch32"
    LARGE_PATCH14 = "large_patch14"
    LARGE_PATCH14_336 = "large_patch14_336"


class ModelLoader(ForgeModel):
    """CLIP model loader implementation for JAX."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_PATCH16: ModelConfig(
            pretrained_model_name="openai/clip-vit-base-patch16",
        ),
        ModelVariant.BASE_PATCH32: ModelConfig(
            pretrained_model_name="openai/clip-vit-base-patch32",
        ),
        ModelVariant.LARGE_PATCH14: ModelConfig(
            pretrained_model_name="openai/clip-vit-large-patch14",
        ),
        ModelVariant.LARGE_PATCH14_336: ModelConfig(
            pretrained_model_name="openai/clip-vit-large-patch14-336",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_PATCH32

    sample_text = ["a photo of a cat", "a photo of a dog"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        Returns:
            ModelInfo: Information about the model and variant
        """
        # Use the provided variant or fall back to default
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="clip",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_CAPT,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the CLIP model instance for this instance's variant.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
        Returns:
            model: The loaded model instance
        """

        from easydel import CLIPModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Check if we need to load from PyTorch weights
        from_pt = pretrained_model_name == "openai/clip-vit-large-patch14-336"

        # Load the model
        model = CLIPModel.from_pretrained(
            pretrained_model_name, from_pt=from_pt, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the CLIP model with this instance's variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.

            mesh: Optional device mesh for sharding (DataParallel mode).
        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        from datasets import load_dataset
        from transformers import AutoImageProcessor

        if mesh is not None:
            # For multi-device, use a fixed batch size that's divisible by device count
            # This matches the original test which used batch_size=8
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 8  # Fixed batch size, will be sharded across devices
            # Ensure batch size is divisible by number of devices
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            # Default to 8 for single device too, for consistency
            batch_size = 8

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        # Load a sample image from Hugging Face cats-image dataset

        dataset = load_dataset("huggingface/cats-image", split="test")
        # Get the first image from the dataset
        image = dataset[0]["image"]

        processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        # Process the inputs
        inputs = processor(
            text=self.sample_text,
            images=image,
            return_tensors="jax",
        )

        input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
        return input_ids

    def get_input_activations_partition_spec(self, mesh, axis_name="X"):
        """Get partition specification for input activations.

        Args:
            mesh: The device mesh for sharding.
            axis_name: Name of the sharding axis.

        Returns:
            PartitionSpec for input activations (sharded on batch dimension)
        """
        if np.prod(list(mesh.shape.values())) == 1:
            return PartitionSpec()

        return PartitionSpec(axis_name)

    def load_parameters_partition_spec(
        self,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        dtype_override=None,
    ):
        # Get the model state
        state = nnx.split(model_for_multichip)[1]

        partition_rules = ((r".*", PartitionSpec()),)  # Everything replicated

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules
        )

    def wrapper_model(self, f):
        """Wrapper for model forward method that extracts the appropriate output.

        CLIP models output both image and text embeddings. This wrapper extracts
        the text model pooler output for compatibility with the testing framework.

        Args:
            f: The model forward function to wrap

        Returns:
            Wrapped function that extracts text model pooler output
        """

        def model(args, kwargs):
            out = f(*args, **kwargs)
            # Extract text model pooler output from the CLIP model output
            out = out.text_model_output.pooler_output
            return out

        return model
