# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lightricks/LTX-2 component loader (128x128, 9-frame bringup resolution).

LTX-2 is an audiovisual text/image-to-video diffusion pipeline brought up by
composite components:

  - TextEncoder  -> Gemma3ForConditionalGeneration  (~12B)
  - Connectors   -> LTX2TextConnectors               (~1.4B)
  - Transformer  -> LTX2VideoTransformer3DModel      (~19B, denoiser - must run on device)
  - Vae          -> AutoencoderKLLTX2Video decoder   (~1.2B)
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
    DTYPE,
    MESH_NAMES,
    MESH_SHAPES,
    PROMPT,
    REPO_ID,
    Gemma3TextEncoderWrapper,
    LTX2ConnectorsWrapper,
    LTX2TransformerWrapper,
    LTX2VaeDecoderWrapper,
    load_connectors,
    load_text_encoder,
    load_transformer,
    load_vae,
    make_audio_latents,
    make_caption_embeds,
    make_packed_prompt_embeds,
    make_prompt_attention_mask,
    make_vae_decoder_input,
    make_video_latents,
    shard_text_encoder_specs,
    shard_transformer_specs,
    tokenize_prompt,
)


class ModelVariant(StrEnum):
    """Loadable components of the LTX-2 pipeline."""

    TEXT_ENCODER = "TextEncoder"
    CONNECTORS = "Connectors"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual LTX-2 components without instantiating LTX2Pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.CONNECTORS: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    _TEXT_VARIANTS = (ModelVariant.TEXT_ENCODER,)
    # Components small enough to fit on a single chip.
    _SINGLE_CHIP_VARIANTS = (ModelVariant.VAE, ModelVariant.CONNECTORS)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in cls._TEXT_VARIANTS
            else ModelTask.MM_VIDEO_TTT
        )
        return ModelInfo(
            model="LTX2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return Gemma3TextEncoderWrapper(load_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.CONNECTORS:
            return LTX2ConnectorsWrapper(load_connectors(dtype)).eval()
        if self._variant == ModelVariant.TRANSFORMER:
            return LTX2TransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return LTX2VaeDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        VAE and Connectors fit on a single chip so any count maps to (1, 1).
        The large TextEncoder and Transformer are tensor-parallel on "model".
        """
        if self._variant in self._SINGLE_CHIP_VARIANTS:
            return (1, 1), MESH_NAMES

        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec dict for the active component."""
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model.text_encoder)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return synthetic inputs (positional list) for the active component."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids, attention_mask = tokenize_prompt(PROMPT)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.CONNECTORS:
            return [make_packed_prompt_embeds(dtype), make_prompt_attention_mask()]

        if self._variant == ModelVariant.TRANSFORMER:
            return [
                make_video_latents(dtype),
                make_audio_latents(dtype),
                make_caption_embeds(dtype),
                make_caption_embeds(dtype),
                torch.tensor([500.0], dtype=dtype),
            ]

        if self._variant == ModelVariant.VAE:
            return [make_vae_decoder_input(dtype)]

        raise ValueError(f"Unknown variant: {self._variant}")
