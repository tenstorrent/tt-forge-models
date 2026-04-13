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
    """Available Qwen2.5 model variants for causal language modeling."""

    ZERO_PT_FIVE_B = "0.5B"
    ZERO_PT_FIVE_B_INSTRUCT = "0.5B_Instruct"


class ModelLoader(ForgeModel):
    """Qwen2.5 model loader for causal LM tasks using EasyDeL/JAX."""

    _VARIANTS = {
        ModelVariant.ZERO_PT_FIVE_B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-0.5B",
            max_length=128,
        ),
        ModelVariant.ZERO_PT_FIVE_B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-0.5B-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ZERO_PT_FIVE_B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.input_text = "My name is Qwen and I am a language model"
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen2.5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from easydel import AutoEasyDeLModelForCausalLM

        if self.tokenizer is None:
            self._load_tokenizer()

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
            self._load_tokenizer()

        inputs = self.tokenizer(self.input_text, return_tensors="jax")
        input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
        return {"input_ids": input_ids}

    def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
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
        state = nnx.split(model_for_multichip)[1]

        if (
            parallelism.name == Parallelism.DATA_PARALLEL.name
            or parallelism.name == Parallelism.SINGLE_DEVICE.name
        ):
            partition_rules = ((r".*", PartitionSpec()),)
        else:
            from easydel.modules.qwen2 import Qwen2Config

            partition_rules = Qwen2Config().get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )
