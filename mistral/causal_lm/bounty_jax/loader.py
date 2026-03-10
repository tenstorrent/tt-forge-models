# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Small 3.1 24B model loader for tensor-parallel causal language modeling.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec
from transformers import MistralConfig

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
from .src.model import Axis, MistralModel

_TP_SHARDING_RULES = [
    (Axis.QHEAD, "X"),
    (Axis.KVHEAD, None),  
    (Axis.MLP, "X"),
    (Axis.EMBED, None),
    (Axis.VOCAB, None),
    (Axis.HEAD, None),
]


class ModelVariant(StrEnum):
    """Available Mistral Small 3.1 24B model variants."""

    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


class ModelLoader(ForgeModel):
    """
    Mistral Small 3.1 24B tensor-parallel model loader.
    Intentionally small model for testing purposes.
    """

    _TEST_VOCAB_SIZE = 131072
    _TEST_HIDDEN_SIZE = 512
    _TEST_INTERMEDIATE_SIZE = 1024
    _TEST_NUM_LAYERS = 8
    _TEST_NUM_ATTENTION_HEADS = 8
    _TEST_NUM_KV_HEADS = 2
    _TEST_HEAD_DIM = 64  # hidden_size // num_attention_heads
    _TEST_MAX_POS_EMBEDDINGS = 64
    _TEST_BATCH_SIZE = 1
    _TEST_SEQ_LEN = 32

    _VARIANTS = {
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="mistralai/Mistral-Small-3.1-24B-Base-2503",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="mistralai/Mistral-Small-3.1-24B-Base-2503",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="mistralai/Mistral-Small-3.1-24B-Base-2503",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CUSTOM_1X2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._mistral_config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mistral-Small-3.1-24B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def _get_mistral_config(self):
        if self._mistral_config is None:
            config = MistralConfig(
                vocab_size=self._TEST_VOCAB_SIZE,
                hidden_size=self._TEST_HIDDEN_SIZE,
                intermediate_size=self._TEST_INTERMEDIATE_SIZE,
                num_hidden_layers=self._TEST_NUM_LAYERS,
                num_attention_heads=self._TEST_NUM_ATTENTION_HEADS,
                num_key_value_heads=self._TEST_NUM_KV_HEADS,
                head_dim=self._TEST_HEAD_DIM,
                max_position_embeddings=self._TEST_MAX_POS_EMBEDDINGS,
            )
            config.mesh = None

            def set_model_mesh(mesh):
                config.mesh = mesh

            config.set_model_mesh = set_model_mesh
            self._mistral_config = config
        return self._mistral_config

    def load_model(self, *, dtype_override=None):
        config = self._get_mistral_config()
        return MistralModel(config, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0), sharding_rules=_TP_SHARDING_RULES)

    def load_inputs(self, dtype_override=None, mesh=None):
        """Return inputs as a dict — required for nnx forward pass unpacking."""
        rng = np.random.default_rng(42)
        input_ids = jnp.array(
            rng.integers(1, 1000, size=(self._TEST_BATCH_SIZE, self._TEST_SEQ_LEN), dtype=np.int32)
        )
        return {"input_ids": input_ids}

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
        model = model_for_multichip if model_for_multichip is not None else self.load_model()
        return nnx.split(model)[1]

    def load_parameters_partition_spec(
        self,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        parallelism=None,
        dtype_override=None
    ):
        model = model_for_multichip if model_for_multichip is not None else self.load_model()
        _, state = nnx.split(model)

        rules_dict = {str(axis): mesh_axis for axis, mesh_axis in _TP_SHARDING_RULES}

        def make_spec(variable):
            logical = getattr(variable, "sharding", None)
            if logical is None:
                return PartitionSpec()
            physical = tuple(rules_dict.get(str(ax), None) for ax in logical)
            return PartitionSpec(*physical)

        flat = state.flat_state()
        specs = [make_spec(var) for _, var in flat]
        flat_specs = nnx.graph.FlatState.from_sorted_keys_values(flat.paths, specs)
        return flat_specs.to_nested_state()

    def get_input_activations_partition_spec(self, mesh, axis_name="X", parallelism=None):
        inputs = self.load_inputs()
        return tuple(PartitionSpec() for _ in inputs)
