# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO (Tencent) component loader for text-to-image generation.

SRPO is a FLUX.1-dev fine-tune that ships only the denoising transformer weights
and reuses the rest of the FLUX.1-dev pipeline. This loader exposes each
independently-compilable component as a variant; the Transformer (denoiser) is
the bringup target, the others come straight from FLUX.1-dev for the composite.
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
from .src.model_utils import (
    BASE_REPO,
    DTYPE,
    GUIDANCE_SCALE,
    MAX_SEQUENCE_LENGTH,
    MESH_NAMES,
    MESH_SHAPES,
    SRPO_REPO,
    CLIPTextEncoderWrapper,
    FluxTransformerWrapper,
    T5TextEncoderWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_text_encoder_2,
    load_transformer,
    load_vae,
    make_packed_latents,
    make_pooled_projections,
    make_prompt_embeds,
    prepare_latent_image_ids,
    prepare_text_ids,
    shard_transformer_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the SRPO / FLUX.1-dev pipeline."""

    TRANSFORMER = "transformer"  # SRPO fine-tuned denoiser (bringup target)
    TEXT_ENCODER = "text_encoder"  # CLIP (FLUX.1-dev)
    TEXT_ENCODER_2 = "text_encoder_2"  # T5-XXL (FLUX.1-dev)
    VAE = "vae"  # AutoencoderKL decoder (FLUX.1-dev)


class ModelLoader(ForgeModel):
    """Load individual SRPO pipeline components without instantiating FluxPipeline."""

    _VARIANTS = {
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=SRPO_REPO),
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=BASE_REPO),
        ModelVariant.TEXT_ENCODER_2: ModelConfig(pretrained_model_name=BASE_REPO),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=BASE_REPO),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    _TEXT_VARIANTS = (ModelVariant.TEXT_ENCODER, ModelVariant.TEXT_ENCODER_2)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in cls._TEXT_VARIANTS
            else ModelTask.MM_IMAGE_TTT
        )
        return ModelInfo(
            model="SRPO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            return FluxTransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.TEXT_ENCODER:
            return CLIPTextEncoderWrapper(load_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return T5TextEncoderWrapper(load_text_encoder_2(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Only the transformer is sharded; the small components replicate.
        """
        if self._variant != ModelVariant.TRANSFORMER:
            return (1, 1), MESH_NAMES
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec dict for the active component."""
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return synthetic inputs for the active component at bringup resolution."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            return {
                "hidden_states": make_packed_latents(dtype),
                "encoder_hidden_states": make_prompt_embeds(dtype),
                "pooled_projections": make_pooled_projections(dtype),
                "timestep": torch.tensor([0.5], dtype=dtype),
                "img_ids": prepare_latent_image_ids(dtype),
                "txt_ids": prepare_text_ids(dtype),
                "guidance": torch.tensor([GUIDANCE_SCALE], dtype=dtype),
            }

        if self._variant in self._TEXT_VARIANTS:
            seq = 77 if self._variant == ModelVariant.TEXT_ENCODER else MAX_SEQUENCE_LENGTH
            return {"input_ids": torch.randint(0, 1000, (1, seq), dtype=torch.long)}

        if self._variant == ModelVariant.VAE:
            # FLUX VAE latent channels = 16; bringup latent grid 16x16.
            return {"latents": torch.randn(1, 16, 16, 16, dtype=dtype)}

        raise ValueError(f"Unknown variant: {self._variant}")
