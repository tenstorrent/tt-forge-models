# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 model loader for tt_forge_models.

LTX-2.3 (Lightricks) is a ~22B-parameter DiT-based **audiovisual** foundation
model that jointly generates synchronized video and audio in a single denoiser.
It is distributed as a single native-format ``*.safetensors`` bundle (github
``ltx-core`` layout) that packs the DiT denoiser (~19B), the causal video VAE,
the audio VAE and the vocoder; the Gemma-3-27B text encoder ships separately
(``Lightricks/LTX-2`` diffusers repo, ~100 GB) and is out of scope here.

Repository: https://huggingface.co/Lightricks/LTX-2.3

Components exposed via ``subfolder``:
  - ``transformer``  : LTX2VideoTransformer3DModel — the audiovisual DiT (key)
  - ``vae``          : AutoencoderKLLTX2Video — causal video VAE decoder
  - ``audio_vae``    : AutoencoderKLLTX2Audio — audio VAE

See ``src/utils.py`` for how the native single-file checkpoint is mapped onto
the diffusers 0.38 LTX-2 classes (diffusers has no ``from_pretrained`` for 2.3
yet — "coming soon" per the model card — so we use the single-file converters
plus two small 2.3-vs-2.0 architecture patches).
"""

from typing import Optional

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
    load_audio_vae,
    load_transformer,
    load_transformer_inputs,
    load_vae,
    load_vae_decoder_inputs,
    shard_transformer_specs,
)

SUPPORTED_SUBFOLDERS = {"transformer", "vae", "audio_vae"}


class ModelVariant(StrEnum):
    """Available LTX-2.3 variants."""

    DISTILLED_22B = "22b-distilled"


class ModelLoader(ForgeModel):
    """Loader for the LTX-2.3 audiovisual video-generation pipeline.

    A component is selected with ``subfolder``:
      - ``transformer`` (default): the ~19B audiovisual DiT denoiser (key)
      - ``vae``: causal video VAE decoder
      - ``audio_vae``: audio VAE
    """

    _VARIANTS = {
        ModelVariant.DISTILLED_22B: ModelConfig(
            pretrained_model_name="Lightricks/LTX-2.3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILLED_22B

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: Optional[str] = "transformer",
    ):
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
            model="LTX-2.3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._subfolder == "transformer":
            return load_transformer(dtype)
        elif self._subfolder == "vae":
            return load_vae(dtype)
        elif self._subfolder == "audio_vae":
            return load_audio_vae(dtype)
        raise ValueError(f"Unknown subfolder: {self._subfolder}")

    def load_inputs(self, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._subfolder == "transformer":
            return load_transformer_inputs(dtype)
        elif self._subfolder == "vae":
            return load_vae_decoder_inputs(dtype)
        elif self._subfolder == "audio_vae":
            return load_vae_decoder_inputs(dtype)
        raise ValueError(f"Unknown subfolder: {self._subfolder}")

    def unpack_forward_output(self, output):
        """The DiT returns a ``(video_noise_pred, audio_noise_pred)`` tuple with
        ``return_dict=False``; use the video prediction (primary output) for PCC.
        VAE components return a single tensor / diffusers output handled upstream.
        """
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output

    def get_mesh_config(self, num_devices: int):
        """Return ((batch, model) mesh shape, mesh names) for the active component."""
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec dict for the active component."""
        if self._subfolder == "transformer":
            return shard_transformer_specs(model)
        return None
