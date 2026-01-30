# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi model loader for tt_forge_models.

Mochi is a video generation model by Genmo with ~10B transformer + VAE.
Repository: https://huggingface.co/genmo/mochi-1-preview

Available subfolders:
- vae: AutoencoderKLMochi
- transformer: MochiTransformer3DModel (~10B params, 40.1GB)
- text_encoder: T5EncoderModel (T5-XXL)
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
    load_pipeline,
    load_vae,
    load_transformer,
    load_text_encoder,
    load_vae_encoder_inputs,
    load_vae_decoder_inputs,
    load_transformer_inputs,
    load_text_encoder_inputs,
)

# Supported subfolders for loading individual components
SUPPORTED_SUBFOLDERS = {"vae", "transformer", "text_encoder"}


@dataclass
class MochiConfig(ModelConfig):
    """Configuration for Mochi variants."""

    source: ModelSource
    enable_tiling: bool = False
    tile_sample_min_height: int = 128
    tile_sample_min_width: int = 128
    tile_sample_stride_height: int = 128
    tile_sample_stride_width: int = 128


class ModelVariant(StrEnum):
    """Available Mochi variants."""

    MOCHI = "mochi"
    MOCHI_TILED = "mochi_tiled"


class ModelLoader(ForgeModel):
    """
    Loader for Mochi model.

    Mochi is a video generation model (~10B transformer + VAE). This loader supports:
    - Loading the full pipeline (subfolder=None)
    - Loading specific components via subfolder:
        - 'vae': AutoencoderKLMochi (~362M encoder + 1.45GB decoder)
        - 'transformer': MochiTransformer3DModel (~10B params)
        - 'text_encoder': T5EncoderModel (T5-XXL)

    Variants:
    - MOCHI: Non-tiled mode
    - MOCHI_TILED: Tiled mode (memory efficient for VAE)
    """

    _VARIANTS = {
        ModelVariant.MOCHI: MochiConfig(
            pretrained_model_name="genmo/mochi-1-preview",
            source=ModelSource.HUGGING_FACE,
            enable_tiling=False,
        ),
        ModelVariant.MOCHI_TILED: MochiConfig(
            pretrained_model_name="genmo/mochi-1-preview",
            source=ModelSource.HUGGING_FACE,
            enable_tiling=True,
            tile_sample_min_height=128,
            tile_sample_min_width=128,
            tile_sample_stride_height=128,
            tile_sample_stride_width=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOCHI

    def __init__(
        self, variant: Optional[ModelVariant] = None, subfolder: Optional[str] = None
    ):
        """
        Initialize the model loader.

        Args:
            variant: Model variant to load
            subfolder: Optional subfolder to load specific component:
                - None: Load full MochiPipeline
                - 'vae': Load AutoencoderKLMochi
                - 'transformer': Load MochiTransformer3DModel (~10B params)
                - 'text_encoder': Load T5EncoderModel (T5-XXL)
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
            model="mochi",
            variant=variant,
            group=ModelGroup.PRIORITY,
            task=ModelTask.MM_VIDEO_TTT,  # Video generation task
            source=cls._VARIANTS[variant].source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        config = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder is None:
            return load_pipeline(
                config.pretrained_model_name,
                dtype,
                enable_tiling=config.enable_tiling,
                tile_sample_min_height=config.tile_sample_min_height,
                tile_sample_min_width=config.tile_sample_min_width,
                tile_sample_stride_height=config.tile_sample_stride_height,
                tile_sample_stride_width=config.tile_sample_stride_width,
            )
        elif self._subfolder == "vae":
            return load_vae(
                config.pretrained_model_name,
                dtype,
                enable_tiling=config.enable_tiling,
                tile_sample_min_height=config.tile_sample_min_height,
                tile_sample_min_width=config.tile_sample_min_width,
                tile_sample_stride_height=config.tile_sample_stride_height,
                tile_sample_stride_width=config.tile_sample_stride_width,
            )
        elif self._subfolder == "transformer":
            return load_transformer(config.pretrained_model_name, dtype)
        elif self._subfolder == "text_encoder":
            return load_text_encoder(config.pretrained_model_name, dtype)
        else:
            raise ValueError(f"Unknown subfolder: {self._subfolder}")

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "vae":
            # in kwards we should have vae_type as "decoder" or "encoder"
            if kwargs.get("vae_type") == "decoder":
                return load_vae_decoder_inputs(dtype)
            elif kwargs.get("vae_type") == "encoder":
                return load_vae_encoder_inputs(dtype)
            else:
                raise ValueError(f"Unknown vae_type: {kwargs.get('vae_type')}")
        elif self._subfolder == "transformer":
            return load_transformer_inputs(dtype)
        elif self._subfolder == "text_encoder":
            return load_text_encoder_inputs(dtype)
        else:
            raise RuntimeError(
                "Full pipeline is currently not supported for input loading"
            )

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        """
        Unpack model output to extract tensor.

        Args:
            output: Model forward pass output

        Returns:
            Output tensor
        """
        if hasattr(output, "sample"):
            return output.sample
        elif hasattr(output, "last_hidden_state"):
            # T5EncoderModel output
            return output.last_hidden_state
        elif isinstance(output, tuple):
            return output[0]
        return output
