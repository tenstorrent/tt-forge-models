# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Policy LIBERO Predict2 2B (nvidia/Cosmos-Policy-LIBERO-Predict2-2B) loader.

Cosmos-Policy-LIBERO-Predict2-2B is a 2B-parameter robot manipulation policy model
fine-tuned from Cosmos-Predict2-2B-Video2World. It jointly predicts actions, future
states, and values via unified video diffusion on the LIBERO simulation benchmark.

Repository: https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B
Base model: https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World
"""
from types import SimpleNamespace
from typing import Any, Optional

import torch
from diffusers import Cosmos2VideoToWorldPipeline, CosmosTransformer3DModel
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

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

BASE_MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"
CHECKPOINT_FILENAME = "Cosmos-Policy-LIBERO-Predict2-2B.pt"


class ModelVariant(StrEnum):
    """Available Cosmos Policy LIBERO Predict2 variants."""

    LIBERO_2B = "LIBERO-2B"


class ModelLoader(ForgeModel):
    """
    Loader for NVIDIA Cosmos-Policy-LIBERO-Predict2-2B.

    The checkpoint is a fine-tune of Cosmos-Predict2-2B-Video2World with the same
    DiT architecture. Actions, proprioceptive states, and values are encoded as
    latent frames injected directly into the video diffusion sequence. The
    checkpoint is distributed as a single .pt file that is loaded into the base
    transformer.
    """

    _VARIANTS = {
        ModelVariant.LIBERO_2B: ModelConfig(
            pretrained_model_name="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LIBERO_2B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[Cosmos2VideoToWorldPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Cosmos Policy LIBERO Predict2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> None:
        try:
            self.pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
                BASE_MODEL,
                torch_dtype=dtype,
            )
        except GatedRepoError:
            transformer = CosmosTransformer3DModel().to(dtype)
            self.pipeline = SimpleNamespace(transformer=transformer)

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        checkpoint_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=CHECKPOINT_FILENAME,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        self.pipeline.transformer.load_state_dict(state_dict, strict=False)
        self.pipeline.transformer.eval()
        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        batch_size = 1
        config = self.pipeline.transformer.config

        # Use small latent dimensions for testing
        latent_num_frames = 2
        latent_height = 2
        latent_width = 2

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

        padding_mask = torch.zeros(1, 1, latent_height, latent_width, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "padding_mask": padding_mask,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
