# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO model loader implementation for text-to-image generation.

SRPO (Tencent, arXiv:2509.06942) is a fine-tune of the FLUX.1-dev MM-DiT
transformer (the denoiser). The published checkpoint is *only* the transformer
weights (``diffusion_pytorch_model.safetensors``); inference reuses the rest of
the FLUX.1-dev pipeline (CLIP + T5 text encoders, VAE).

This loader brings up the **denoiser** component: it builds a
``FluxTransformer2DModel`` from the FLUX.1-dev transformer config and loads the
SRPO weights into it. ``load_inputs`` produces correctly-shaped latent / text /
guidance tensors for a single denoising step, so the denoiser can be compiled
and PCC-checked on device without first running the (separately bringable)
text encoders.
"""
import torch
from typing import Optional

from diffusers import FluxTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SRPO model variants."""

    SRPO = "SRPO"


class ModelLoader(ForgeModel):
    """SRPO (FLUX.1-dev denoiser fine-tune) loader for text-to-image generation."""

    # The SRPO checkpoint ships only the transformer weights; the architecture /
    # config comes from FLUX.1-dev.
    _BASE_PIPELINE = "black-forest-labs/FLUX.1-dev"

    _VARIANTS = {
        ModelVariant.SRPO: ModelConfig(
            pretrained_model_name="tencent/SRPO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SRPO

    # Native generation settings (FLUX.1-dev / SRPO defaults from the model card).
    NATIVE_HEIGHT = 1024
    NATIVE_WIDTH = 1024
    NATIVE_MAX_SEQUENCE_LENGTH = 512
    GUIDANCE_SCALE = 3.5

    # Resolution used for the on-device denoiser bring-up gate. Smaller than the
    # native 1024x1024 to keep the single-forward compile/run within the bring-up
    # time budget; the composite pipeline generates at native resolution.
    GATE_HEIGHT = 512
    GATE_WIDTH = 512

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model info for the given variant."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SRPO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the SRPO denoiser (FLUX transformer + SRPO weights).

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype (defaults to bfloat16 for device execution).

        Returns:
            torch.nn.Module: The SRPO FluxTransformer2DModel denoiser.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Build the FLUX transformer architecture from the base pipeline config,
        # without instantiating random weights twice (config only -> small).
        config = FluxTransformer2DModel.load_config(
            self._BASE_PIPELINE, subfolder="transformer"
        )
        model = FluxTransformer2DModel.from_config(config)

        # Load the SRPO transformer weights (published as a single safetensors).
        weights_path = hf_hub_download(
            self._variant_config.pretrained_model_name,
            "diffusion_pytorch_model.safetensors",
        )
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)

        model = model.to(dtype).eval()
        self.model = model
        self._dtype = dtype
        return model

    def _build_inputs(self, height, width, max_sequence_length, dtype, batch_size=1):
        """Construct a single-step set of denoiser inputs at the given resolution.

        The text/pooled embeddings are random tensors of the correct shape/dtype
        (the CLIP/T5 encoders are separate components); this is sufficient to
        compile and PCC-check the denoiser, which compares identical CPU vs device
        inputs.
        """
        # FLUX packs the 8x-downsampled VAE latent into 2x2 patches, so each
        # token covers a 16px image region and carries in_channels (=64) features.
        in_channels = self.model.config.in_channels  # 64
        joint_attention_dim = self.model.config.joint_attention_dim  # 4096 (T5)
        pooled_dim = self.model.config.pooled_projection_dim  # 768 (CLIP)

        latent_h = height // 16
        latent_w = width // 16
        img_seq = latent_h * latent_w

        hidden_states = torch.randn(batch_size, img_seq, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, pooled_dim, dtype=dtype)
        timestep = torch.tensor([1.0] * batch_size, dtype=dtype)

        guidance = None
        if getattr(self.model.config, "guidance_embeds", False):
            guidance = torch.full([batch_size], self.GUIDANCE_SCALE, dtype=dtype)

        # Positional ids for text and image tokens (FLUX RoPE).
        txt_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)
        img_ids = torch.zeros(latent_h, latent_w, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(latent_h)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(latent_w)[None, :]
        img_ids = img_ids.reshape(-1, 3).to(dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_projections,
            "encoder_hidden_states": encoder_hidden_states,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            "joint_attention_kwargs": {},
        }

    def load_inputs(self, dtype_override=None, batch_size=1, height=None, width=None):
        """Return sample inputs for the SRPO denoiser (single denoising step).

        Defaults to the bring-up gate resolution (512x512); pass
        ``height``/``width`` to generate inputs at another resolution (e.g. the
        native 1024x1024 used by the composite pipeline).
        """
        if self.model is None:
            self.load_model(dtype_override=dtype_override)
        dtype = dtype_override if dtype_override is not None else self._dtype

        height = height if height is not None else self.GATE_HEIGHT
        width = width if width is not None else self.GATE_WIDTH

        return self._build_inputs(
            height=height,
            width=width,
            max_sequence_length=self.NATIVE_MAX_SEQUENCE_LENGTH,
            dtype=dtype,
            batch_size=batch_size,
        )
