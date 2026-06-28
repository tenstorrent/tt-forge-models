# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 model loader for tt_forge_models.

LTX-2 (Lightricks/LTX-2) is an audiovisual text/image-to-video diffusion
pipeline built around a ~19B audiovisual DiT denoiser. Like the other diffusion
loaders in this repo, it is brought up by *component* rather than as one graph,
since the scheduler / denoising loop / latent glue stay in host Python.

Components (select via ``subfolder``):
- ``transformer``   : LTX2VideoTransformer3DModel (~19B) -- the denoiser / sharding target
- ``vae``           : AutoencoderKLLTX2Video (conv3d video VAE; decode/encode)
- ``text_encoder``  : Gemma3ForConditionalGeneration (~12B caption encoder)
- ``connectors``    : LTX2TextConnectors (caption -> video/audio cross-attn spaces)
- ``audio_vae``     : AutoencoderKLLTX2Audio (audio latent <-> mel VAE)
- ``None``          : the full LTX2Pipeline (host-Python composite only)

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
    NUM_ATTENTION_HEADS,
    load_audio_vae,
    load_connectors,
    load_pipeline,
    load_text_encoder,
    load_transformer,
    load_transformer_inputs,
    load_text_encoder_inputs,
    load_vae,
    load_vae_decoder_inputs,
    load_vae_encoder_inputs,
    shard_transformer_specs,
)

# Components that can be loaded individually.
SUPPORTED_SUBFOLDERS = {
    "transformer",
    "vae",
    "text_encoder",
    "connectors",
    "audio_vae",
}


class ModelVariant(StrEnum):
    """Available LTX-2 variants."""

    LTX_2_19B = "19b"


class ModelLoader(ForgeModel):
    """Loader for the LTX-2 audiovisual video diffusion pipeline.

    Pass ``subfolder`` to load and test a single component on device; pass
    ``None`` to load the full host-Python pipeline for composite generation.
    """

    _VARIANTS = {
        ModelVariant.LTX_2_19B: ModelConfig(
            pretrained_model_name="Lightricks/LTX-2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LTX_2_19B

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: Optional[str] = None,
    ):
        """Initialize the loader.

        Args:
            variant: Model variant to load.
            subfolder: Component to load (see ``SUPPORTED_SUBFOLDERS``), or None
                for the full pipeline.
        """
        super().__init__(variant)
        if subfolder is not None and subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder

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

    def load_model(self, dtype_override=None, **kwargs):
        # The denoiser & VAE ship in bf16; default to bf16 for device parity, but
        # honor an explicit override (e.g. fp32 for a high-precision CPU golden).
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder is None:
            return load_pipeline(dtype)
        elif self._subfolder == "transformer":
            return load_transformer(dtype)
        elif self._subfolder == "vae":
            return load_vae(dtype)
        elif self._subfolder == "audio_vae":
            return load_audio_vae(dtype)
        elif self._subfolder == "text_encoder":
            return load_text_encoder(dtype)
        elif self._subfolder == "connectors":
            return load_connectors(dtype)
        else:
            raise ValueError(f"Unknown subfolder: {self._subfolder}")

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "transformer":
            return load_transformer_inputs(dtype)
        elif self._subfolder == "vae":
            if kwargs.get("vae_type") == "encoder":
                return load_vae_encoder_inputs(dtype)
            # Default to decoder inputs (the composite VAE op of interest).
            return load_vae_decoder_inputs(dtype)
        elif self._subfolder == "text_encoder":
            return load_text_encoder_inputs(dtype)
        else:
            raise RuntimeError(
                f"load_inputs is not defined for subfolder={self._subfolder!r}"
            )

    def get_mesh_config(self, num_devices: int):
        """Return ((batch, model) mesh shape, mesh names) for the active component.

        The transformer denoiser is the sharding target; text encoder / VAE /
        connectors run replicated on a single chip.
        """
        if self._subfolder != "transformer":
            return (1, 1), MESH_NAMES
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec dict for the transformer denoiser.

        Only valid when the model attention heads divide the model mesh axis.
        """
        if self._subfolder != "transformer":
            return None
        return shard_transformer_specs(model)

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        """Extract a primary tensor from a component's forward output."""
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output
