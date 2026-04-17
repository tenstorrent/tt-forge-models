# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
OpenELM model loader implementation for causal language modeling.
"""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    Parallelism,
    StrEnum,
)
from ....tools.jax_utils import cast_hf_model_to_type
import flax.nnx as nnx
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import numpy as np


class ModelVariant(StrEnum):
    """Available OpenELM model variants."""

    _270M = "270M"
    _450M = "450M"
    _1_1B = "1_1B"
    _3B = "3B"


class ModelLoader(ForgeModel):
    """OpenELM model loader implementation for causal language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant._270M: LLMModelConfig(
            pretrained_model_name="apple/OpenELM-270M",
        ),
        ModelVariant._450M: LLMModelConfig(
            pretrained_model_name="apple/OpenELM-450M",
        ),
        ModelVariant._1_1B: LLMModelConfig(
            pretrained_model_name="apple/OpenELM-1_1B",
        ),
        ModelVariant._3B: LLMModelConfig(
            pretrained_model_name="apple/OpenELM-3B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant._270M

    sample_text = "Hello there fellow traveler"

    # OpenELM does not bundle a tokenizer; use LLaMA tokenizer from a public repo
    _TOKENIZER_SOURCE = "hf-internal-testing/llama-tokenizer"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="OpenELM",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.

        Returns:
            tokenizer: The loaded tokenizer instance
        """

        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._TOKENIZER_SOURCE, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OpenELM model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """

        from easydel import AutoEasyDeLModelForCausalLM

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        partition_rules = ((r".*", PartitionSpec()),)

        # Load the model
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            partition_rules=partition_rules,
            **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the OpenELM model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
            mesh: Optional device mesh for sharding (DataParallel mode).

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        if mesh is not None:
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 8
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            batch_size = 8

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
        return {"input_ids": input_ids}

    def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
        """Get partition specification for input activations.

        Args:
            mesh: The device mesh for sharding.
            parallelism: The level of parallelism for sharding.
            axis_name: The name of the mesh axis to use for sharding.

        Returns:
            PartitionSpec for input activations (sharded on batch dimension)
        """

        if (
            parallelism.name == Parallelism.TENSOR_PARALLEL.name
            or np.prod(list(mesh.shape.values())) == 1
        ):
            return (PartitionSpec(),)

        return (PartitionSpec(axis_name),)

    def load_parameters_partition_spec(
        self,
        model_for_multichip,
        parallelism,
        axis_name="X",
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        dtype_override=None,
    ):
        # Get the model state
        state = nnx.split(model_for_multichip)[1]

        if (
            parallelism.name == Parallelism.DATA_PARALLEL.name
            or parallelism.name == Parallelism.SINGLE_DEVICE.name
        ):
            partition_rules = ((r".*", PartitionSpec()),)
        else:
            # Use EasyDeL's OpenELMConfig to get proper partition rules
            from easydel.modules.openelm import OpenELMConfig

            openelm_config = OpenELMConfig()
            partition_rules = openelm_config.get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )
