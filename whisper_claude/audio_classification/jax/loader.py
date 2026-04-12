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
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    Parallelism,
    StrEnum,
)

# Whisper mel spectrogram constants
_NUM_MEL_BINS = 80
_NUM_FRAMES = 3000


class ModelVariant(StrEnum):
    """Available Whisper model variants for JAX ASR."""

    BASE = "Base"
    MEDIUM = "Medium"
    LARGE_V3 = "Large_v3"


class ModelLoader(ForgeModel):
    """Whisper JAX encoder-decoder model loader using EasyDeL.

    Whisper is an encoder-decoder architecture. Both encoder and decoder inputs
    are partitioned independently, so get_input_activations_partition_spec
    returns two PartitionSpec values — one per input tensor:
      [0] input_features  (encoder): mel spectrogram (batch, num_mel_bins, num_frames)
      [1] decoder_input_ids (decoder): token IDs     (batch, decoder_seq_len)

    Partition strategy:
      - TENSOR_PARALLEL:  both inputs replicated, weights sharded via WhisperConfig rules
      - DATA_PARALLEL:    both inputs batch-sharded, weights fully replicated
      - SINGLE_DEVICE:    both inputs replicated, weights fully replicated (1-device mesh)
    """

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="openai/whisper-base",
        ),
        ModelVariant.MEDIUM: ModelConfig(
            pretrained_model_name="openai/whisper-medium",
        ),
        ModelVariant.LARGE_V3: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Whisper",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from easydel import AutoEasyDeLModelForSpeechSeq2Seq

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        partition_rules = ((r".*", PartitionSpec()),)
        model = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
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

        dtype = dtype_override if dtype_override is not None else jnp.float32

        # Encoder input: mel spectrogram features (batch, num_mel_bins, num_frames)
        input_features = jnp.zeros(
            (batch_size, _NUM_MEL_BINS, _NUM_FRAMES), dtype=dtype
        )

        # Decoder input: single forced BOS token to seed generation
        decoder_input_ids = jnp.zeros((batch_size, 1), dtype=jnp.int32)

        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
        }

    # ------------------------------------------------------------------
    # Multi-chip partition methods
    # ------------------------------------------------------------------

    def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
        """Return partition specs for encoder and decoder inputs.

        Whisper has two distinct inputs that must be partitioned independently:
          - input_features:    mel spectrogram fed to the encoder
          - decoder_input_ids: token IDs fed to the decoder

        For TENSOR_PARALLEL (or single-device mesh) both are replicated so
        every device runs the full forward pass on the same data while weights
        are split. For DATA_PARALLEL both are batch-sharded so each device
        processes a distinct slice of the batch.
        """
        if (
            parallelism.name == Parallelism.TENSOR_PARALLEL.name
            or np.prod(list(mesh.shape.values())) == 1
        ):
            # Both encoder and decoder inputs replicated
            return (PartitionSpec(), PartitionSpec())
        # Both encoder and decoder inputs batch-sharded for data parallelism
        return (PartitionSpec(axis_name), PartitionSpec(axis_name))

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
        - TENSOR_PARALLEL: parameters sharded according to WhisperConfig's
          built-in partition rules, which split attention heads in both the
          encoder and decoder across the model axis.
        """
        state = nnx.split(model_for_multichip)[1]

        if (
            parallelism.name == Parallelism.DATA_PARALLEL.name
            or parallelism.name == Parallelism.SINGLE_DEVICE.name
        ):
            partition_rules = ((r".*", PartitionSpec()),)
        else:
            from easydel.modules.whisper import WhisperConfig

            partition_rules = WhisperConfig().get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )
