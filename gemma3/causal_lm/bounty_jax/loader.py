# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 model loader for tensor-parallel causal language modeling.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.model import Gemma3Config, Gemma3ForCausalLM


class ModelVariant(StrEnum):
    """Available Gemma3 model variants."""

    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


class ModelLoader(ForgeModel):
    """
    Gemma3 tensor-parallel model loader.
    Intentionally small model for testing purposes.
    """

    _TEST_VOCAB_SIZE = 32000
    _TEST_HIDDEN_SIZE = 256
    _TEST_INTERMEDIATE_SIZE = 512
    _TEST_NUM_LAYERS = 6
    _TEST_NUM_ATTENTION_HEADS = 8
    _TEST_NUM_KV_HEADS = 4
    _TEST_HEAD_DIM = 32
    _TEST_MAX_POS_EMBEDDINGS = 64
    _TEST_SLIDING_WINDOW = 32
    _TEST_SLIDING_WINDOW_PATTERN = 6
    _TEST_BATCH_SIZE = 1
    _TEST_SEQ_LEN = 32

    _VARIANTS = {
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="google/gemma-3-4b",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="google/gemma-3-4b",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="google/gemma-3-4b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CUSTOM_1X2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._gemma_config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Gemma3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def _get_gemma_config(self):
        if self._gemma_config is None:
            self._gemma_config = Gemma3Config(
                vocab_size=self._TEST_VOCAB_SIZE,
                hidden_size=self._TEST_HIDDEN_SIZE,
                intermediate_size=self._TEST_INTERMEDIATE_SIZE,
                num_hidden_layers=self._TEST_NUM_LAYERS,
                num_attention_heads=self._TEST_NUM_ATTENTION_HEADS,
                num_key_value_heads=self._TEST_NUM_KV_HEADS,
                head_dim=self._TEST_HEAD_DIM,
                max_position_embeddings=self._TEST_MAX_POS_EMBEDDINGS,
                sliding_window=self._TEST_SLIDING_WINDOW,
                sliding_window_pattern=self._TEST_SLIDING_WINDOW_PATTERN,
                use_cache=False,
            )
        return self._gemma_config

    def load_model(self, *, dtype_override=None, **_):
        config = self._get_gemma_config()
        if dtype_override is not None:
            config.dtype = dtype_override
            config.param_dtype = dtype_override
        return Gemma3ForCausalLM(config, rngs=nnx.Rngs(0))

    def load_inputs(self, dtype_override=None, mesh=None, **_):
        """Return inputs as a dict — required for nnx forward pass unpacking."""
        rng = np.random.default_rng(42)
        input_ids = jnp.array(
            rng.integers(1, 1000, size=(self._TEST_BATCH_SIZE, self._TEST_SEQ_LEN), dtype=np.int32)
        )
        position_ids = jnp.broadcast_to(
            jnp.arange(self._TEST_SEQ_LEN)[None, :], (self._TEST_BATCH_SIZE, self._TEST_SEQ_LEN)
        )
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
        }

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
        **_,
    ):
        model = model_for_multichip if model_for_multichip is not None else self.load_model()
        return nnx.split(model)[1]

    def load_parameters_partition_spec(
        self,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        parallelism=None,
        dtype_override=None,
        **_,
    ):
        model = model_for_multichip if model_for_multichip is not None else self.load_model()
        state = nnx.split(model)[1]
        return jax.tree.map(lambda _: PartitionSpec(), state)

    def get_input_activations_partition_spec(self, mesh, axis_name="X", parallelism=None, **_):
        inputs = self.load_inputs()
        return tuple(PartitionSpec() for _ in inputs)
