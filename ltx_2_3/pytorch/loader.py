# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 model loader for tt_forge_models.

LTX-2.3 is Lightricks' ~22B DiT joint audio-video foundation model
(`Lightricks/LTX-2.3`). It generates synchronized video and audio in a single
model. The checkpoint is a single bundled native safetensors file containing the
transformer, a video VAE, an audio VAE, a vocoder, and in-checkpoint text
connectors. See `src/utils.py` for how this loader bridges the native checkpoint
to the diffusers `LTX2VideoTransformer3DModel` / `AutoencoderKLLTX2Video`
classes (diffusers has no LTX-2.3 `from_pretrained` path yet).

Components (one loader per independently-compilable component, diffusion style):
- 'transformer': LTX2VideoTransformer3DModel (~22B params) — the denoiser.
- 'vae':         AutoencoderKLLTX2Video — the video VAE.
"""

from dataclasses import dataclass
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
    DEV_CKPT,
    DISTILLED_CKPT,
    MESH_NAMES,
    MESH_SHAPES,
    load_transformer,
    load_transformer_inputs,
    load_video_vae,
    shard_transformer_specs,
)

SUPPORTED_SUBFOLDERS = {"transformer", "vae"}


@dataclass
class LTX23Config(ModelConfig):
    """Configuration for LTX-2.3 variants."""

    source: ModelSource = ModelSource.HUGGING_FACE
    checkpoint: str = DISTILLED_CKPT


class ModelVariant(StrEnum):
    """Available LTX-2.3 variants."""

    DISTILLED = "distilled"
    DEV = "dev"


class ModelLoader(ForgeModel):
    """Loader for the LTX-2.3 joint audio-video model.

    Variants:
    - DISTILLED: ltx-2.3-22b-distilled (8 steps, CFG=1) — the default.
    - DEV:       ltx-2.3-22b-dev (full, trainable bf16).

    Components (via `subfolder`):
    - 'transformer': LTX2VideoTransformer3DModel (~22B) — the denoiser/sharding target.
    - 'vae':         AutoencoderKLLTX2Video — the video VAE.
    """

    _VARIANTS = {
        ModelVariant.DISTILLED: LTX23Config(
            pretrained_model_name="Lightricks/LTX-2.3",
            checkpoint=DISTILLED_CKPT,
        ),
        ModelVariant.DEV: LTX23Config(
            pretrained_model_name="Lightricks/LTX-2.3",
            checkpoint=DEV_CKPT,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILLED

    def __init__(
        self, variant: Optional[ModelVariant] = None, subfolder: Optional[str] = None
    ):
        """Initialize the loader.

        Args:
            variant: Model variant to load (DISTILLED or DEV).
            subfolder: Component to load — 'transformer' (default) or 'vae'.
        """
        super().__init__(variant)
        if subfolder is None:
            subfolder = "transformer"
        if subfolder not in SUPPORTED_SUBFOLDERS:
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
            task=ModelTask.MM_VIDEO_TTT,  # joint audio-video generation
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, **kwargs):
        config = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._subfolder == "transformer":
            return load_transformer(dtype, checkpoint=config.checkpoint)
        elif self._subfolder == "vae":
            return load_video_vae(dtype, checkpoint=config.checkpoint)
        raise ValueError(f"Unknown subfolder: {self._subfolder}")

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._subfolder == "transformer":
            return load_transformer_inputs(dtype)
        raise RuntimeError(
            f"Input loading not implemented for subfolder={self._subfolder}"
        )

    def get_mesh_config(self, num_devices: int):
        """Return ((batch, model) mesh shape, mesh names) for the transformer."""
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec dict for the transformer (Megatron)."""
        if self._subfolder == "transformer":
            return shard_transformer_specs(model)
        return None

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        """Unpack model output to a tensor (video noise prediction)."""
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
