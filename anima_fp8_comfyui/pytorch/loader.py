# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Anima FP8 ComfyUI model loader implementation.

Loads the FP8-quantized single-file Anima text-to-image diffusion transformer
from pachiiahri/anima-fp8-comfyui. Anima is a 2B parameter anime/illustration
text-to-image model released by CircleStone Labs, architecturally derived from
NVIDIA's Cosmos-Predict2-2B-Text2Image. This checkpoint uses ComfyUI's
TensorCoreFP8Layout hardware-fp8 quantization.

Available variants:
- ANIMA_PREVIEW_TCFP8_MIXED: Mixed-precision hardware FP8 preview checkpoint.
"""

from typing import Any, Optional

import torch
from diffusers import CosmosTransformer3DModel
from huggingface_hub import hf_hub_download

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

FP8_REPO_ID = "pachiiahri/anima-fp8-comfyui"
UPSTREAM_CONFIG_REPO = "nvidia/Cosmos-Predict2-2B-Text2Image"


class ModelVariant(StrEnum):
    """Available Anima FP8 ComfyUI model variants."""

    ANIMA_PREVIEW_TCFP8_MIXED = "Preview-TCFP8-Mixed"


class ModelLoader(ForgeModel):
    """Anima FP8 ComfyUI model loader."""

    _VARIANTS = {
        ModelVariant.ANIMA_PREVIEW_TCFP8_MIXED: ModelConfig(
            pretrained_model_name=FP8_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANIMA_PREVIEW_TCFP8_MIXED

    FP8_FILES = {
        ModelVariant.ANIMA_PREVIEW_TCFP8_MIXED: "anima-preview_tcfp8_mixed.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ANIMA_FP8_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> CosmosTransformer3DModel:
        """Load the FP8-quantized Anima diffusion transformer from a single safetensors file."""
        fp8_file = self.FP8_FILES[self._variant]
        fp8_path = hf_hub_download(repo_id=FP8_REPO_ID, filename=fp8_file)
        self._transformer = CosmosTransformer3DModel.from_single_file(
            fp8_path,
            config=UPSTREAM_CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FP8-quantized Anima diffusion transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare synthetic inputs for the Anima diffusion transformer."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = 1

        if self._transformer is None:
            self._load_transformer(dtype)

        config = self._transformer.config

        # Small latent dimensions keep the forward pass cheap for testing.
        latent_num_frames = 1
        latent_height = 16
        latent_width = 16

        in_channels = config.in_channels
        hidden_states = torch.randn(
            batch_size,
            in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        text_embed_dim = config.text_embed_dim
        encoder_hidden_states = torch.randn(batch_size, 8, text_embed_dim, dtype=dtype)

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
