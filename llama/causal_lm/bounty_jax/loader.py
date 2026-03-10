# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.1 8B model loader for tensor-parallel causal language modeling.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen

from tt_forge_models.base import ForgeModel
from tt_forge_models.config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.config import LLaMAConfig
from .src.model import FlaxLLaMAForCausalLMModule


class _LlamaWrapper(linen.Module):
    """ Unpacks the (input_ids, attention_mask, position_ids) tuple """

    config: Any
    dtype: Any

    @linen.compact
    def __call__(self, inputs):
        input_ids, attention_mask, position_ids = inputs
        module = FlaxLLaMAForCausalLMModule(
            config=self.config, dtype=self.dtype
        )
        return module(input_ids, attention_mask, position_ids)


class ModelVariant(StrEnum):
    """ Available Llama 3.1 8B model variants """

    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


class ModelLoader(ForgeModel):
    """ 
        Llama 3.1 8B tensor-parallel model loader 
        Intentionally small model for testing purposes.
    """
    
    _TEST_VOCAB_SIZE = 128256
    _TEST_HIDDEN_SIZE = 512
    _TEST_INTERMEDIATE_SIZE = 1024
    _TEST_NUM_LAYERS = 4
    _TEST_NUM_ATTENTION_HEADS = 8
    _TEST_NUM_KV_HEADS = 2
    _TEST_MAX_SEQ_LEN = 64
    _TEST_BATCH_SIZE = 1
    _TEST_SEQ_LEN = 32

    _VARIANTS = {
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-8B",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-8B",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CUSTOM_1X2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._llama_config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Llama3.1-8B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def _get_llama_config(self):
        if self._llama_config is None:
            self._llama_config = LLaMAConfig(
                vocab_size=self._TEST_VOCAB_SIZE,
                hidden_size=self._TEST_HIDDEN_SIZE,
                intermediate_size=self._TEST_INTERMEDIATE_SIZE,
                num_hidden_layers=self._TEST_NUM_LAYERS,
                num_attention_heads=self._TEST_NUM_ATTENTION_HEADS,
                num_key_value_heads=self._TEST_NUM_KV_HEADS,
                max_sequence_length=self._TEST_MAX_SEQ_LEN,
            )
        return self._llama_config

    def load_model(self, *, dtype_override=None):
        if dtype_override is not None:
            dtype = dtype_override
        else:
            dtype = jnp.bfloat16
    
        return _LlamaWrapper(config=self._get_llama_config(), dtype=dtype)

    def load_inputs(self, dtype_override=None, mesh=None):
        """
            Return (input_ids, attention_mask, position_ids) as a single tuple.

            Passed as one activation argument to model.apply, _LlamaWrapper unpacks it.
        """
        batch_size = self._TEST_BATCH_SIZE
        seq_len = self._TEST_SEQ_LEN
        rng = np.random.default_rng(42)
        input_ids = jnp.array(
            rng.integers(1, 1000, size=(batch_size, seq_len), dtype=np.int32)
        )
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        position_ids = jnp.broadcast_to(
            jnp.arange(seq_len)[None, :], (batch_size, seq_len)
        )
        return (input_ids, attention_mask, position_ids)

    def load_parameters(
        self,
        dtype_override=None,
        train=False,
        seed=None,
        inputs=None,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        input_parameters_partition_specs=None,
    ):
        from infra.utilities import initialize_flax_linen_parameters_on_cpu

        if inputs is None:
            inputs = self.load_inputs(mesh=cpu_mesh)
        model = (
            model_for_multichip
            if model_for_multichip is not None
            else self.load_model(dtype_override=dtype_override)
        )
        return initialize_flax_linen_parameters_on_cpu(
            model,
            input_activations_partition_specs,
            inputs,
            input_parameters_partition_specs,
            cpu_mesh,
            seed if seed is not None else 42,
        )

    def load_parameters_partition_spec(
        self,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        dtype_override=None,
        parallelism=None,
    ):
        from infra.utilities import make_flax_linen_parameters_partition_specs_on_cpu

        if inputs is None:
            inputs = self.load_inputs(mesh=cpu_mesh)
        model = (
            model_for_multichip
            if model_for_multichip is not None
            else self.load_model(dtype_override=dtype_override)
        )
        return make_flax_linen_parameters_partition_specs_on_cpu(
            model, cpu_mesh, input_activations_partition_specs, inputs
        )

    def get_input_activations_partition_spec(self, mesh, axis_name="X", parallelism=None):
        from jax.sharding import PartitionSpec
        
        return (PartitionSpec(),)
