# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-chip utilities for AlexNet JAX model initialization.
"""

from typing import Any

import jax
from flax import linen
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec


def make_flax_linen_parameters_partition_specs_on_cpu(
    model: linen.Module,
    cpu_mesh: Mesh,
    inputs_specs: PartitionSpec,
    cpu_inputs: Any
):
    """Make partition specs for Flax linen model parameters on CPU.

    This function determines how model parameters should be partitioned
    across devices for multi-chip configurations.

    Args:
        model: The Flax linen model
        cpu_mesh: JAX Mesh object for CPU devices
        inputs_specs: Partition specifications for the input tensors
        cpu_inputs: Input tensors on CPU

    Returns:
        Partition specifications for the model parameters
    """
    # Use shard_map because CCL ops need a mapped axis for tracing to work
    return linen.get_partition_spec(
        jax.eval_shape(
            shard_map(
                model.init,
                cpu_mesh,
                in_specs=(None, inputs_specs),
                out_specs=PartitionSpec(),
            ),
            jax.random.PRNGKey(21),
            cpu_inputs,
        )
    )


def initialize_flax_linen_parameters_on_cpu(
    model: linen.Module,
    inputs_specs: PartitionSpec,
    cpu_inputs: Any,
    params_specs: Any,
    cpu_mesh: Mesh,
    rng_seed: int,
):
    """Initialize Flax linen model parameters on CPU for multi-chip configurations.

    This function uses JAX's shard_map to properly initialize model parameters
    across multiple devices with the specified partitioning.

    Args:
        model: The Flax linen model to initialize
        inputs_specs: Partition specifications for the input tensors
        cpu_inputs: Input tensors on CPU
        params_specs: Partition specifications for the model parameters
        cpu_mesh: JAX Mesh object for CPU devices
        rng_seed: Random seed for parameter initialization

    Returns:
        Initialized model parameters with proper sharding
    """
    init_fn = shard_map(
        model.init, cpu_mesh, in_specs=(None, inputs_specs), out_specs=params_specs
    )
    return init_fn(jax.random.PRNGKey(rng_seed), cpu_inputs)