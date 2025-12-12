# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon model loader implementation for causal language modeling
"""
from typing import Optional
from transformers import AutoTokenizer

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel

import flax.nnx as nnx
from jax.sharding import PartitionSpec
import numpy as np


class ModelVariant(StrEnum):
    """Available Falcon model variants."""

    FALCON_1B = "tiiuae/Falcon3-1B-Base"
    FALCON_3B = "tiiuae/Falcon3-3B-Base"
    FALCON_7B = "tiiuae/Falcon3-7B-Base"
    FALCON_10B = "tiiuae/Falcon3-10B-Base"
    FALCON_MAMBA_7B = "tiiuae/Falcon3-Mamba-7B-Base"
    FALCON_7B_INSTRUCT = "tiiuae/falcon-7b-instruct"


class ModelLoader(ForgeModel):
    """Falcon model loader implementation for causal LM tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.FALCON_1B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-1B-Base",
        ),
        ModelVariant.FALCON_3B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-3B-Base",
        ),
        ModelVariant.FALCON_7B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-7B-Base",
        ),
        ModelVariant.FALCON_10B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-10B-Base",
        ),
        ModelVariant.FALCON_MAMBA_7B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-Mamba-7B-Base",
        ),
        ModelVariant.FALCON_7B_INSTRUCT: ModelConfig(
            pretrained_model_name="tiiuae/falcon-7b-instruct",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.FALCON_1B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant in [
            ModelVariant.FALCON_1B,
            ModelVariant.FALCON_3B,
            ModelVariant.FALCON_7B,
            ModelVariant.FALCON_10B,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="falcon",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        super().__init__(variant)

        # Configuration parameters
        self.input_text_1 = "Write a function to calculate the factorial of a number"
        self.max_length = 512
        self.tokenizer = None
        self.input_text_2 = "Hello, my dog is cute"
        self._model_name = self._variant_config.pretrained_model_name

    def load_model(self, dtype_override=None):
        """Load and return the Falcon model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Falcon model instance for causal LM.
        """
        from easydel import AutoEasyDeLModelForCausalLM

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            self._model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the Falcon model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        if self._variant == ModelVariant.FALCON_7B_INSTRUCT:
            inputs = self.tokenizer(self.input_text_2, return_tensors="jax").input_ids
        else:
            inputs = self.tokenizer.encode(
                self.input_text_1,
                add_special_tokens=True,
                return_tensors="jax",
                max_length=self.max_length,
                truncation=True,
            )
        return inputs

    def get_mesh_config(self, num_devices: int):
        # Single-device: always return data-parallel (1, 1)
        if num_devices == 1:
            return (1, 1), ("batch", "model")

        # Variant-specific mesh decisions for clarity and robustness
        if self._variant == ModelVariant.FALCON_MAMBA_7B:
            assert (
                num_devices % 2 == 0
            ), "Mamba requires an even number of devices for (2, N/2) mesh"
            mesh_shape = (2, num_devices // 2)
            return mesh_shape, ("batch", "model")

        # All other Falcon variants have attention heads in config
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        else:
            assert num_devices % 2 == 0, "Attention heads cannot be evenly distributed"
            mesh_shape = (2, num_devices // 2)

        shard_attention = self._variant in [
            ModelVariant.FALCON_7B,
            ModelVariant.FALCON_10B,
        ]
        if shard_attention:
            assert (
                self.config.num_attention_heads % mesh_shape[1] == 0
            ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

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
