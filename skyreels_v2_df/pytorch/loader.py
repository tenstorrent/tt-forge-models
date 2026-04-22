# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SkyReels-V2 Diffusion Forcing model loader for tt_forge_models.

SkyReels-V2 is an autoregressive Diffusion Forcing Transformer for long-form
video generation. It supports text-to-video and image-to-video synthesis with
per-token noise levels, enabling synchronous or asynchronous denoising
schedules across frames.

Repository:
- https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
"""

from typing import Any, Optional

import torch
from diffusers import SkyReelsV2DiffusionForcingPipeline  # type: ignore[import]

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


class ModelVariant(StrEnum):
    """Available SkyReels-V2 Diffusion Forcing variants."""

    SKYREELS_V2_DF_1_3B_540P = "1.3B_540P"


class ModelLoader(ForgeModel):
    """SkyReels-V2 Diffusion Forcing model loader for text-to-video generation."""

    _VARIANTS = {
        ModelVariant.SKYREELS_V2_DF_1_3B_540P: ModelConfig(
            pretrained_model_name="Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SKYREELS_V2_DF_1_3B_540P

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[SkyReelsV2DiffusionForcingPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SkyReelsV2_DF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> SkyReelsV2DiffusionForcingPipeline:
        self.pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the SkyReels-V2 Diffusion Forcing transformer.

        Returns:
            SkyReelsV2Transformer3DModel (torch.nn.Module) ready for inference.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        return self.pipeline.transformer

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare synthetic inputs for the SkyReels-V2 transformer forward pass."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        return self._load_transformer_inputs(dtype)

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        batch_size = 1
        config = self.pipeline.transformer.config

        # patch_size is [p_t, p_h, p_w]; spatial dims must be divisible by p_h, p_w
        p_h, p_w = config.patch_size[1], config.patch_size[2]
        in_channels = config.in_channels  # 16
        text_dim = config.text_dim  # 4096

        latent_frames = 1
        latent_height = p_h * 2  # 4, minimal valid spatial size
        latent_width = p_w * 2  # 4

        hidden_states = torch.randn(
            batch_size,
            in_channels,
            latent_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )
        timestep = torch.randint(0, 1000, (batch_size,), dtype=torch.long)
        encoder_hidden_states = torch.randn(batch_size, 8, text_dim, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "fps": [0] * batch_size,  # fps category index: 0=16fps, 1=other
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
