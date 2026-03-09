# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon3 7B model loader for tensor-parallel causal language modeling.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen

from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...base import ForgeModel
from .src.config import FalconConfig
from .src.model import FlaxFalconForCausalLMModule


class _FalconWrapper(linen.Module):
    """Unpacks the (input_ids, attention_mask, position_ids) tuple. """

    config: Any
    dtype: Any

    @linen.compact
    def __call__(self, inputs):
        input_ids, attention_mask, position_ids = inputs
        inner = FlaxFalconForCausalLMModule(config=self.config, dtype=self.dtype)
        return inner(input_ids, attention_mask, position_ids)


class ModelVariant(StrEnum):
    """Available Falcon3 7B model variants."""

    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


class ModelLoader(ForgeModel):
    """ 
        Falcon3 7B tensor-parallel model loader 
        Intentionally small model for testing purposes.
    """

    _TEST_VOCAB_SIZE = 131072
    _TEST_HIDDEN_SIZE = 512
    _TEST_INTERMEDIATE_SIZE = 1024
    _TEST_NUM_LAYERS = 2
    _TEST_NUM_ATTENTION_HEADS = 8
    _TEST_NUM_KV_HEADS = 2
    _TEST_MAX_SEQ_LEN = 64
    _TEST_BATCH_SIZE = 1
    _TEST_SEQ_LEN = 32

    _VARIANTS = {
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-7B-Base",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-7B-Base",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-7B-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CUSTOM_1X2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._falcon_config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Falcon3-7B",
            variant=variant,
            group=ModelGroup.GENERALITY, # TODO: depends on test type
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def _get_falcon_config(self):
        if self._falcon_config is None:
            self._falcon_config = FalconConfig(
                vocab_size=self._TEST_VOCAB_SIZE,
                hidden_size=self._TEST_HIDDEN_SIZE,
                intermediate_size=self._TEST_INTERMEDIATE_SIZE,
                num_hidden_layers=self._TEST_NUM_LAYERS,
                num_attention_heads=self._TEST_NUM_ATTENTION_HEADS,
                num_key_value_heads=self._TEST_NUM_KV_HEADS,
                max_sequence_length=self._TEST_MAX_SEQ_LEN,
            )
        return self._falcon_config

    def load_model(self, *, dtype_override=None, **_):
        dtype = dtype_override if dtype_override is not None else jnp.bfloat16
        return _FalconWrapper(config=self._get_falcon_config(), dtype=dtype)

    def load_inputs(self, **_):
        """
            Return (input_ids, attention_mask, position_ids) as a single tuple.

            Passed as one activation argument to model.apply, _FalconWrapper unpacks it.
        """
        rng = np.random.default_rng(42)
        input_ids = jnp.array(
            rng.integers(1, 1000, size=(self._TEST_BATCH_SIZE, self._TEST_SEQ_LEN), dtype=np.int32)
        )
        attention_mask = jnp.ones((self._TEST_BATCH_SIZE, self._TEST_SEQ_LEN), dtype=jnp.int32)
        position_ids = jnp.broadcast_to(
            jnp.arange(self._TEST_SEQ_LEN)[None, :], (self._TEST_BATCH_SIZE, self._TEST_SEQ_LEN)
        )
        return (input_ids, attention_mask, position_ids)

    def load_parameters(self, *, seed=None, inputs=None, model_for_multichip=None, **_):
        if inputs is None:
            inputs = self.load_inputs()

        if model_for_multichip is not None:
            model = model
        else:
            model = self.load_model()

        with jax.default_device(jax.devices("cpu")[0]):
            return model.init(jax.random.PRNGKey(seed if seed is not None else 42), inputs)

    def load_parameters_partition_spec(self, *, inputs=None, model_for_multichip=None, **_):
        from jax.sharding import PartitionSpec

        if inputs is None:
            inputs = self.load_inputs()

        if model_for_multichip is not None:
            model = model
        else:
            model = self.load_model()

        with jax.default_device(jax.devices("cpu")[0]):
            params = model.init(jax.random.PRNGKey(42), inputs)

        return jax.tree.map(lambda _: PartitionSpec(), params)

    def get_input_activations_partition_spec(self, *_, **__):
        from jax.sharding import PartitionSpec

        return (tuple(PartitionSpec() for _ in self.load_inputs()),)
