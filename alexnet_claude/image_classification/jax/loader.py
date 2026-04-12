# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.sharding import PartitionSpec

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    Parallelism,
    StrEnum,
)

# Standard ImageNet input dimensions (JAX channels-last convention)
_IMAGE_H = 224
_IMAGE_W = 224
_IMAGE_C = 3
_NUM_CLASSES = 1000


class AlexNet(nn.Module):
    """AlexNet image classification model in Flax Linen.

    Follows the standard AlexNet topology:
      Conv(96, 11×11, s4) → Pool → Conv(256, 5×5) → Pool →
      Conv(384, 3×3) → Conv(384, 3×3) → Conv(256, 3×3) → Pool →
      Dense(4096) → Dense(4096) → Dense(num_classes)

    Input layout: channels-last (batch, H, W, C) per JAX/Flax convention.
    """

    num_classes: int = _NUM_CLASSES

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=96, kernel_size=(11, 11), strides=(4, 4), padding=0)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        x = nn.Conv(features=256, kernel_size=(5, 5), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        x = nn.Conv(features=384, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=384, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=4096)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4096)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


class ModelVariant(StrEnum):
    """Available AlexNet JAX variants; name encodes the target mesh topology."""

    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


class ModelLoader(ForgeModel):
    """AlexNet custom Flax Linen loader with tensor parallelism support.

    AlexNet is a non-EasyDeL Flax Linen model. Partitioning uses the
    infra.utilities Flax Linen helpers:
      - initialize_flax_linen_parameters_on_cpu  — initialise params on CPU mesh
      - make_flax_linen_parameters_partition_specs_on_cpu — assign PartitionSpec

    All weights are replicated across devices (AlexNet has no attention heads
    to shard), so partitioning is uniform `((r".*", PartitionSpec()),)`.
    Activations are replicated for TENSOR_PARALLEL and batch-sharded for
    DATA_PARALLEL.
    """

    _VARIANTS = {
        ModelVariant.CUSTOM_1X2: ModelConfig(pretrained_model_name=""),
        ModelVariant.CUSTOM_1X4: ModelConfig(pretrained_model_name=""),
        ModelVariant.CUSTOM_1X8: ModelConfig(pretrained_model_name=""),
    }

    DEFAULT_VARIANT = ModelVariant.CUSTOM_1X2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AlexNet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        return AlexNet()

    def load_inputs(self, dtype_override=None, mesh=None):
        if mesh is not None:
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 8
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            batch_size = 8

        dtype = dtype_override if dtype_override is not None else jnp.float32
        images = jnp.zeros((batch_size, _IMAGE_H, _IMAGE_W, _IMAGE_C), dtype=dtype)
        return (images,)

    # ------------------------------------------------------------------
    # Multi-chip partition methods
    # ------------------------------------------------------------------

    def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
        """Return partition spec for the single image input tensor.

        AlexNet has one input (images). For TENSOR_PARALLEL or single-device
        mesh the input is replicated; for DATA_PARALLEL it is batch-sharded.
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
        """Return partition specs for all AlexNet parameters.

        AlexNet has no attention heads or MLP intermediate dims to shard, so
        all parameters are replicated uniformly regardless of parallelism mode.
        The Flax Linen infra utilities handle parameter initialisation on the
        CPU mesh and annotate every leaf with a replicated PartitionSpec.
        """
        from infra.utilities import (
            initialize_flax_linen_parameters_on_cpu,
            make_flax_linen_parameters_partition_specs_on_cpu,
        )

        init_params = initialize_flax_linen_parameters_on_cpu(
            model=model_for_multichip, cpu_mesh=cpu_mesh, inputs=inputs
        )
        return make_flax_linen_parameters_partition_specs_on_cpu(
            params=init_params, partition_rules=((r".*", PartitionSpec()),)
        )
