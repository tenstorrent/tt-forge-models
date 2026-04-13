# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    Parallelism,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Falcon3 model variants for JAX causal LM."""

    FALCON3_1B = "3_1B_Base"


class ModelLoader(ForgeModel):
    """Falcon3-1B causal LM loader for JAX using EasyDeL.

    Supports TENSOR_PARALLEL, DATA_PARALLEL, and SINGLE_DEVICE modes.

    Partition strategy:
      - TENSOR_PARALLEL:  inputs replicated, weights sharded via FalconConfig rules
      - DATA_PARALLEL:    inputs batch-sharded, weights fully replicated
      - SINGLE_DEVICE:    inputs replicated, weights fully replicated (1-device mesh)
    """

    _VARIANTS = {
        ModelVariant.FALCON3_1B: LLMModelConfig(
            pretrained_model_name="tiiuae/Falcon3-1B-Base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON3_1B

    sample_text = "What is the capital of France?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Falcon3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from easydel import AutoEasyDeLModelForCausalLM

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        partition_rules = ((r".*", PartitionSpec()),)
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            self._model_name, partition_rules=partition_rules, **model_kwargs
        )
        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        if mesh is not None:
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 8
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            batch_size = 8

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self.tokenizer(self.sample_text, return_tensors="jax")
        input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
        return {"input_ids": input_ids}

    def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
        """Return partition specs for input activations.

        - TENSOR_PARALLEL or single-device mesh: inputs replicated across all
          devices — each device sees the full batch.
        - DATA_PARALLEL: inputs sharded along the batch axis so each device
          processes a distinct slice.
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
        """Return partition specs for all model parameters.

        - DATA_PARALLEL / SINGLE_DEVICE: all parameters replicated.
        - TENSOR_PARALLEL: parameters sharded according to FalconConfig's
          built-in partition rules.
        """
        state = nnx.split(model_for_multichip)[1]

        if (
            parallelism.name == Parallelism.DATA_PARALLEL.name
            or parallelism.name == Parallelism.SINGLE_DEVICE.name
        ):
            partition_rules = ((r".*", PartitionSpec()),)
        else:
            from easydel.modules.falcon import FalconConfig

            partition_rules = FalconConfig().get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )
