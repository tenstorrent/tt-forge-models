# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.1 8B model loader for tensor-parallel causal language modeling.

Single-chip Llama is covered by the EasyDel loader in
third_party/tt_forge_models/llama/causal_lm/jax/loader.py.
"""

import importlib.util as _ilu
import os
import sys
from typing import Any, Optional

# ---------------------------------------------------------------------------
# src/ files use bare imports (e.g. `from config import LLaMAConfig`) and
# call jax.devices("cpu") at module level.  We need two things:
#
# 1. _SRC_DIR in sys.path so bare imports resolve correctly.
# 2. Lazy loading via _ensure_src_loaded() so module-level JAX ops don't run
#    during pytest collection (before the jax_num_cpu_devices fixture fires).
#
# We also use importlib.util.spec_from_file_location with unique names to
# avoid sys.modules collisions with unrelated packages (e.g. torchxrayvision
# has its own 'model' and 'config' packages already cached).
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen

# Absolute imports — bypass the broken "llama3.1_8b" package hierarchy.
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

_llama_model_module = None
_llama_config_module = None


def _load_src(unique_name, filename):
    spec = _ilu.spec_from_file_location(unique_name, os.path.join(_SRC_DIR, filename))
    mod = _ilu.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_src_loaded():
    global _llama_model_module, _llama_config_module
    if _llama_model_module is None:
        _llama_config_module = _load_src("_llama3_1_8b_config", "config.py")
        _llama_model_module = _load_src("_llama3_1_8b_model", "model.py")


class _LlamaWrapper(linen.Module):
    """Thin linen.Module that unpacks the (input_ids, attention_mask, position_ids) tuple.

    The tester always calls model.apply(params, activations) where activations is
    whatever load_inputs() returns as a single object.  Our load_inputs() returns a
    3-tuple, so Flax's apply would pass the whole tuple as one positional arg to
    __call__, which expects three separate args.  This wrapper unpacks the tuple.
    """

    config: Any
    dtype: Any

    @linen.compact
    def __call__(self, inputs):
        input_ids, attention_mask, position_ids = inputs
        inner = _llama_model_module.FlaxLLaMAForCausalLMModule(
            config=self.config, dtype=self.dtype
        )
        return inner(input_ids, attention_mask, position_ids)


class ModelVariant(StrEnum):
    """Available Llama 3.1 8B model variants."""

    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


_VARIANT_NUM_DEVICES = {
    ModelVariant.CUSTOM_1X2: 2,
    ModelVariant.CUSTOM_1X4: 4,
    ModelVariant.CUSTOM_1X8: 8,
}


class ModelLoader(ForgeModel):
    """Llama 3.1 8B tensor-parallel model loader."""

    _TEST_VOCAB_SIZE = 128256
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
        _ensure_src_loaded()
        if self._llama_config is None:
            self._llama_config = _llama_config_module.LLaMAConfig(
                vocab_size=self._TEST_VOCAB_SIZE,
                hidden_size=self._TEST_HIDDEN_SIZE,
                intermediate_size=self._TEST_INTERMEDIATE_SIZE,
                num_hidden_layers=self._TEST_NUM_LAYERS,
                num_attention_heads=self._TEST_NUM_ATTENTION_HEADS,
                num_key_value_heads=self._TEST_NUM_KV_HEADS,
                max_sequence_length=self._TEST_MAX_SEQ_LEN,
            )
        return self._llama_config

    def _make_wrapper(self, dtype):
        return _LlamaWrapper(config=self._get_llama_config(), dtype=dtype)

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_src_loaded()
        dtype = dtype_override if dtype_override is not None else jnp.bfloat16
        return self._make_wrapper(dtype)

    def load_inputs(self, dtype_override=None, mesh=None):
        """Return (input_ids, attention_mask, position_ids) as a single tuple.

        Passed as one activation argument to model.apply; _LlamaWrapper unpacks it.
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

    def load_multichip_model(
        self, axis_name="X", num_devices=2, train_mode=False, dtype_override=None
    ):
        _ensure_src_loaded()
        dtype = dtype_override if dtype_override is not None else jnp.bfloat16
        tt_devices = jax.devices()[:num_devices]
        _llama_model_module.mesh = jax.sharding.Mesh(
            np.array(tt_devices).reshape(num_devices), axis_names=("mp",)
        )
        return self._make_wrapper(dtype)

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
        if inputs is None:
            inputs = self.load_inputs(dtype_override)
        model = (
            model_for_multichip
            if model_for_multichip is not None
            else self.load_model(dtype_override=dtype_override)
        )
        with jax.default_device(jax.devices("cpu")[0]):
            return model.init(
                jax.random.PRNGKey(seed if seed is not None else 42),
                inputs,
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
        from jax.sharding import PartitionSpec

        if inputs is None:
            inputs = self.load_inputs(dtype_override)
        model = (
            model_for_multichip
            if model_for_multichip is not None
            else self.load_model(dtype_override=dtype_override)
        )
        with jax.default_device(jax.devices("cpu")[0]):
            params = model.init(jax.random.PRNGKey(42), inputs)
        return jax.tree.map(lambda _: PartitionSpec(), params)

    def get_input_activations_partition_spec(self, mesh, axis_name="X", parallelism=None):
        from jax.sharding import PartitionSpec

        inputs = self.load_inputs()
        # Return a 1-element outer tuple containing the nested pytree spec for the
        # inputs tuple.  After *-unpacking in _get_forward_method_arg_specs:
        #   in_specs = (params_spec, (PS(), PS(), PS()))  — 2 top-level items,
        # matching the 2 positional args: model.apply(params, inputs_tuple).
        return (tuple(PartitionSpec() for _ in inputs),)
