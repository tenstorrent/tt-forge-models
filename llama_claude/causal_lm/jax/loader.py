# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from ....base import ForgeModel
from ....config import (
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
    """Available Llama model variants for JAX causal LM."""

    LLAMA_1B_TINY = "1B_Tiny"
    LLAMA_3B_V2 = "3B_v2"


class ModelLoader(ForgeModel):
    """Llama JAX model loader for causal LM using EasyDeL.

    Supports TENSOR_PARALLEL, DATA_PARALLEL, and SINGLE_DEVICE modes.

    Partition strategy:
      - TENSOR_PARALLEL:  inputs replicated, weights sharded via LlamaConfig rules
      - DATA_PARALLEL:    inputs batch-sharded, weights fully replicated
      - SINGLE_DEVICE:    inputs replicated, weights fully replicated (1-device mesh)
    """

    _VARIANTS = {
        ModelVariant.LLAMA_1B_TINY: LLMModelConfig(
            pretrained_model_name="TinyLlama/TinyLlama_v1.1",
            max_length=128,
        ),
        ModelVariant.LLAMA_3B_V2: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-3B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_1B_TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.input_text = "Hey how are you doing today?"
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Llama",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )
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

        inputs = self.tokenizer(self.input_text, return_tensors="jax")
        input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
        return {"input_ids": input_ids}

    # ------------------------------------------------------------------
    # Multi-chip partition methods (required even for single-device mesh
    # because DynamicJaxMultiChipModelTester is always used for EasyDeL)
    # ------------------------------------------------------------------

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
        - TENSOR_PARALLEL: parameters sharded according to LlamaConfig's
          built-in partition rules, which split attention heads and MLP
          intermediate dimensions across the model axis.
        """
        state = nnx.split(model_for_multichip)[1]

        if (
            parallelism.name == Parallelism.DATA_PARALLEL.name
            or parallelism.name == Parallelism.SINGLE_DEVICE.name
        ):
            partition_rules = ((r".*", PartitionSpec()),)
        else:
            from easydel.modules.llama import LlamaConfig

            partition_rules = LlamaConfig().get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )
