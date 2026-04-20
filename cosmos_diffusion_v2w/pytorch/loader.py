# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos 1.0 Diffusion 14B Video2World model loader for tt_forge_models.

Cosmos Video2World is a 14B parameter diffusion-based world foundation model by NVIDIA
that generates physics-aware future video frames from an input video (or image) plus a
text prompt.

Repository: https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-14B-Video2World

Available subfolders:
- transformer: The diffusion transformer (DiT) denoiser
- vae: Video tokenizer (encoder/decoder)
- text_encoder: Text encoder for prompt conditioning
"""

import logging
from typing import Any, Optional

import torch
from diffusers import CosmosVideoToWorldPipeline
from diffusers.models.autoencoders.autoencoder_kl_cosmos import AutoencoderKLCosmos
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from diffusers.schedulers.scheduling_edm_euler import EDMEulerScheduler
from transformers import T5Config, T5EncoderModel, T5Tokenizer

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

logger = logging.getLogger(__name__)

SUPPORTED_SUBFOLDERS = {"transformer", "vae", "text_encoder"}


class ModelVariant(StrEnum):
    """Available Cosmos Diffusion Video2World variants."""

    V1_14B = "1.0-14B"


class _NoOpSafetyChecker:
    """Placeholder when cosmos_guardrail is not installed."""

    pass


class ModelLoader(ForgeModel):
    """
    Loader for Cosmos 1.0 Diffusion 14B Video2World model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': The diffusion transformer denoiser
    - 'vae': Video tokenizer (AutoencoderKL)
    - 'text_encoder': T5-based text encoder
    """

    _VARIANTS = {
        ModelVariant.V1_14B: ModelConfig(
            pretrained_model_name="nvidia/Cosmos-1.0-Diffusion-14B-Video2World",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_14B

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: Optional[str] = None,
    ):
        super().__init__(variant)
        if subfolder is not None and subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder
        self.pipeline: Optional[CosmosVideoToWorldPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Cosmos Diffusion Video2World",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline_from_config(self, dtype: torch.dtype):
        """Build the pipeline from default configs (no pretrained download)."""
        transformer = CosmosTransformer3DModel()
        vae = AutoencoderKLCosmos()
        t5_config = T5Config(
            d_model=1024,
            num_heads=16,
            d_ff=2816,
            num_layers=2,
            decoder_start_token_id=0,
        )
        text_encoder = T5EncoderModel(t5_config)
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        scheduler = EDMEulerScheduler()

        self.pipeline = CosmosVideoToWorldPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            safety_checker=_NoOpSafetyChecker(),
        )
        self.pipeline.transformer = self.pipeline.transformer.to(dtype=dtype)
        self.pipeline.vae = self.pipeline.vae.to(dtype=dtype)
        self.pipeline.text_encoder = self.pipeline.text_encoder.to(dtype=dtype)

    def _load_pipeline(
        self, dtype: torch.dtype, **kwargs
    ) -> CosmosVideoToWorldPipeline:
        model_kwargs = {"torch_dtype": dtype}
        model_kwargs |= kwargs
        try:
            self.pipeline = CosmosVideoToWorldPipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                **model_kwargs,
            )
        except OSError:
            logger.warning(
                "Cannot access %s (gated repo), using config-only init",
                self._variant_config.pretrained_model_name,
            )
            self._load_pipeline_from_config(dtype)
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype, **kwargs)

        if self._subfolder == "vae":
            return self.pipeline.vae
        elif self._subfolder == "text_encoder":
            return self.pipeline.text_encoder
        elif self._subfolder == "transformer" or self._subfolder is None:
            return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "transformer" or self._subfolder is None:
            return self._load_transformer_inputs(dtype)
        elif self._subfolder == "vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_vae_decoder_inputs(dtype)
            else:
                return self._load_vae_encoder_inputs(dtype)
        elif self._subfolder == "text_encoder":
            return self._load_text_encoder_inputs(dtype)

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        batch_size = 1
        config = self.pipeline.transformer.config

        num_latent_frames = 2
        latent_height = 2
        latent_width = 2

        in_channels = config.in_channels
        hidden_states = torch.randn(
            batch_size,
            in_channels,
            num_latent_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        enc_dim = getattr(
            config, "joint_attention_dim", config.encoder_hidden_states_channels
        )
        encoder_hidden_states = torch.randn(batch_size, 8, enc_dim, dtype=dtype)

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        padding_mask = torch.zeros(1, 1, latent_height, latent_width, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "padding_mask": padding_mask,
            "return_dict": False,
        }

    def _load_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        latent_channels = self.pipeline.vae.config.latent_channels
        return {
            "sample": torch.randn(1, latent_channels, 2, 2, 2, dtype=dtype),
        }

    def _load_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        return {
            "sample": torch.randn(1, 3, 9, 64, 64, dtype=dtype),
        }

    def _load_text_encoder_inputs(self, dtype: torch.dtype) -> dict:
        return {
            "input_ids": torch.randint(0, 1000, (1, 16)),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output
