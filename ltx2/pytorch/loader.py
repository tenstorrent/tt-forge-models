# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LTX-2 (Lightricks) model loader for tt_forge_models.

LTX-2 is a joint audio-visual video generation pipeline (`LTX2Pipeline`,
diffusers). Like other diffusion bringups it is decomposed into independently
compilable components, exposed here as variants so the runner discovers one
device test per component:

  - ``transformer``  : LTX2VideoTransformer3DModel (~19B audio-visual DiT denoiser)
  - ``vae_decoder``  : AutoencoderKLLTX2Video decoder (video latents -> RGB frames)

The denoiser is the heavy sharding target (37.8 GB bf16 > 32 GB/chip), so it
ships TP shard hooks (``get_mesh_config`` / ``load_shard_spec``). The VAE decoder
fits a single chip.

Repository: https://huggingface.co/Lightricks/LTX-2
"""

from typing import Any, Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.utils import (
    MESH_NAMES,
    MESH_SHAPES,
    PRETRAINED,
    load_transformer,
    load_transformer_inputs,
    load_vae_decoder,
    load_vae_decoder_inputs,
    shard_transformer_specs,
    unpack_transformer_output,
)


class ModelVariant(StrEnum):
    """Available LTX-2 components (one independently-compilable network each)."""

    TRANSFORMER = "transformer"
    VAE_DECODER = "vae_decoder"


class ModelLoader(ForgeModel):
    """Loader for LTX-2 video generation components.

    Variants:
      - TRANSFORMER: LTX2VideoTransformer3DModel (~19B), the audio-visual
        denoiser and the gate for the pipeline. Sharded across chips.
      - VAE_DECODER: AutoencoderKLLTX2Video decoder, single chip.
    """

    _VARIANTS = {
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=PRETRAINED),
        ModelVariant.VAE_DECODER: ModelConfig(pretrained_model_name=PRETRAINED),
    }

    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX-2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer(dtype)
        elif self._variant == ModelVariant.VAE_DECODER:
            return load_vae_decoder(dtype)
        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer_inputs(dtype)
        elif self._variant == ModelVariant.VAE_DECODER:
            return load_vae_decoder_inputs(dtype)
        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """TP mesh for the active component.

        The transformer shards across all chips on the ("batch", "model") mesh;
        the VAE decoder runs single-chip.
        """
        if self._variant == ModelVariant.VAE_DECODER:
            return (1, 1), MESH_NAMES
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Tensor -> partition_spec dict for the active component.

        Only the transformer is sharded (Megatron column->row over attention and
        FFN). The VAE decoder runs replicated on a single chip.
        """
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model)
        return None

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if self._variant == ModelVariant.TRANSFORMER:
            return unpack_transformer_output(output)
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
