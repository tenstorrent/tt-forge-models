# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2-dev component loader (black-forest-labs/FLUX.2-dev).

Each variant corresponds to one independently loadable component of the
text-to-image pipeline:
  - Dev             -> Flux2Transformer2DModel  (MM-DiT, ~32 B)      [tensor parallel]
  - Dev_TextEncoder -> Mistral3 text encoder    (~24 B)              [tensor parallel]
  - Dev_VAE         -> AutoencoderKLFlux2 decoder (~0.1 B)           [single device]

The transformer and text encoder are too large for a single chip and ship with
SPMD shard specs (see src/model_utils.py). The full end-to-end image generation
(text encoder -> DiT denoise -> VAE decode) is exposed via `generate_image`,
used to produce a sample image as a bringup artifact.
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
    DIT_IN_CHANNELS,
    DIT_JOINT_ATTENTION_DIM,
    DIT_NUM_ROPE_AXES,
    DTYPE,
    FLUX2_REPO_ID,
    IMAGE_SEQ_LEN,
    LATENT_HW,
    MESH_NAMES,
    MESH_SHAPES,
    TEXT_SEQ_LEN,
    TEXT_VOCAB_SIZE,
    VAE_LATENT_CHANNELS,
    Flux2TransformerWrapper,
    Flux2VAEDecoderWrapper,
    Mistral3PromptEmbedWrapper,
    load_text_encoder,
    load_transformer,
    load_vae,
    shard_flux2_transformer_specs,
    shard_mistral3_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the FLUX.2-dev pipeline."""

    DEV = "Dev"  # transformer (MM-DiT)
    DEV_TEXT_ENCODER = "Dev_TextEncoder"
    DEV_VAE = "Dev_VAE"


class ModelLoader(ForgeModel):
    """Load individual FLUX.2-dev components without pulling the full pipeline."""

    _VARIANTS = {
        ModelVariant.DEV: ModelConfig(pretrained_model_name=FLUX2_REPO_ID),
        ModelVariant.DEV_TEXT_ENCODER: ModelConfig(pretrained_model_name=FLUX2_REPO_ID),
        ModelVariant.DEV_VAE: ModelConfig(pretrained_model_name=FLUX2_REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.DEV

    # Variants that require tensor parallelism (do not fit on a single chip).
    _SHARDED_VARIANTS = (ModelVariant.DEV, ModelVariant.DEV_TEXT_ENCODER)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        if variant == ModelVariant.DEV_TEXT_ENCODER:
            task = ModelTask.NLP_EMBED_GEN
        else:
            task = ModelTask.MM_IMAGE_TTT
        return ModelInfo(
            model="FLUX.2-dev",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module."""
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.DEV:
            return Flux2TransformerWrapper(load_transformer(model_name, dtype))
        if self._variant == ModelVariant.DEV_TEXT_ENCODER:
            return Mistral3PromptEmbedWrapper(load_text_encoder(model_name, dtype))
        if self._variant == ModelVariant.DEV_VAE:
            return Flux2VAEDecoderWrapper(load_vae(model_name, dtype))

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of positional input tensors for the active component."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.DEV:
            hidden_states = torch.randn(1, IMAGE_SEQ_LEN, DIT_IN_CHANNELS, dtype=dtype)
            encoder_hidden_states = torch.randn(
                1, TEXT_SEQ_LEN, DIT_JOINT_ATTENTION_DIM, dtype=dtype
            )
            timestep = torch.tensor([1.0], dtype=dtype)
            img_ids = self._build_latent_ids()
            txt_ids = self._build_text_ids()
            guidance = torch.full((1,), 4.0, dtype=dtype)
            return [
                hidden_states,
                encoder_hidden_states,
                timestep,
                img_ids,
                txt_ids,
                guidance,
            ]

        if self._variant == ModelVariant.DEV_TEXT_ENCODER:
            input_ids = torch.randint(
                0, TEXT_VOCAB_SIZE, (1, TEXT_SEQ_LEN), dtype=torch.long
            )
            attention_mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.DEV_VAE:
            z = torch.randn(
                1, VAE_LATENT_CHANNELS, LATENT_HW, LATENT_HW, dtype=dtype
            )
            return [z]

        raise ValueError(f"Unknown variant: {self._variant}")

    @staticmethod
    def _build_latent_ids():
        """4D (T, H, W, L) coords for packed image tokens (mirrors pipeline)."""
        h = LATENT_HW // 2
        w = LATENT_HW // 2
        coords = torch.cartesian_prod(
            torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1)
        )
        return coords.to(torch.float32)  # (IMAGE_SEQ_LEN, 4)

    @staticmethod
    def _build_text_ids():
        """4D coords for text tokens: only the sequence axis varies."""
        coords = torch.cartesian_prod(
            torch.arange(1),
            torch.arange(1),
            torch.arange(1),
            torch.arange(TEXT_SEQ_LEN),
        )
        return coords.to(torch.float32)  # (TEXT_SEQ_LEN, 4)

    # ------------------------------------------------------------------
    # Multi-chip (tensor parallel) support
    # ------------------------------------------------------------------

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Sharded variants (transformer, text encoder) map to (1, num_devices).
        The VAE always runs single-chip -> (1, 1).
        """
        if self._variant not in self._SHARDED_VARIANTS:
            return (1, 1), MESH_NAMES
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec dict for the active component.

        Expects the wrapper module returned by load_model().
        """
        if self._variant == ModelVariant.DEV:
            return shard_flux2_transformer_specs(model.transformer)
        if self._variant == ModelVariant.DEV_TEXT_ENCODER:
            return shard_mistral3_specs(model.text_encoder)
        return None

    # ------------------------------------------------------------------
    # End-to-end image generation (bringup artifact)
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str = "A photorealistic astronaut riding a horse on the moon, "
        "earth in the sky, cinematic lighting",
        *,
        num_inference_steps: int = 8,
        guidance_scale: float = 4.0,
        height: int = 256,
        width: int = 256,
        seed: int = 0,
        dtype_override: Optional[torch.dtype] = None,
    ):
        """Run the full FLUX.2-dev pipeline (text encoder -> DiT -> VAE) on CPU
        and return a PIL image. Exercises all three components together."""
        from diffusers import Flux2Pipeline

        dtype = dtype_override if dtype_override is not None else DTYPE
        pipe = Flux2Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype
        )
        pipe.to("cpu")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        return result.images[0]
