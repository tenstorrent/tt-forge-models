# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 component loader for tt_forge_models.

LTX-2.3 (`Lightricks/LTX-2.3`) is a ~22B-parameter DiT-based joint audio-video
generation model from Lightricks. Like other diffusion pipelines it is brought
up by independently-compilable component, one ``ModelVariant`` each:

  - TRANSFORMER  : LTX2VideoTransformer3DModel  (~22B, the per-step denoiser)
  - VAE          : AutoencoderKLLTX2Video        (latent <-> pixel video)
  - TEXT_ENCODER : Gemma3ForConditionalGeneration (prompt encoder)

See ``src/utils.py`` for how the 2.3 single-file checkpoint is loaded and why the
VAE / text-encoder / transformer-config are sourced from the architecture-
identical ``Lightricks/LTX-2`` diffusers release.
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
from .src.utils import (
    DTYPE,
    LTX23_REPO,
    AUDIO_NUM_FRAMES,
    LATENT_H,
    LATENT_T,
    LATENT_W,
    MESH_NAMES,
    MESH_SHAPES,
    LTX2TransformerWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_text_encoder_inputs,
    load_transformer,
    load_transformer_inputs,
    load_vae,
    load_vae_decoder_inputs,
    load_vae_encoder_inputs,
    shard_transformer_specs,
)


class ModelVariant(StrEnum):
    """Independently-loadable components of the LTX-2.3 pipeline."""

    TRANSFORMER = "transformer"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"


class ModelLoader(ForgeModel):
    """Load individual LTX-2.3 components without instantiating the full pipeline."""

    _VARIANTS = {
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=LTX23_REPO),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=LTX23_REPO),
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=LTX23_REPO),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.MM_VIDEO_TTT
        )
        return ModelInfo(
            model="LTX-2.3",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        use_reference_weights: bool = False,
        **kwargs,
    ):
        """Load and return the component for this variant as a torch.nn.Module.

        Args:
            dtype_override: weight dtype (defaults to bfloat16).
            use_reference_weights: TRANSFORMER only - load the architecture-
                identical LTX-2 transformer instead of the 22B 2.3 single-file
                checkpoint (see ``src/utils.load_transformer``).
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            transformer = load_transformer(
                dtype, use_reference_weights=use_reference_weights
            )
            return LTX2TransformerWrapper(
                transformer,
                num_frames=LATENT_T,
                height=LATENT_H,
                width=LATENT_W,
                audio_num_frames=AUDIO_NUM_FRAMES,
            ).eval()
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(dtype)).eval()
        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return ((batch, model) mesh shape, mesh names) for TP sharding.

        The transformer is the sharding target; VAE / text-encoder stay single-chip.
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return {param_tensor: partition_spec} for the TRANSFORMER variant.

        ``model`` is the ``LTX2TransformerWrapper`` returned by ``load_model``;
        the specs are derived from its wrapped ``.transformer``.
        """
        if self._variant == ModelVariant.TRANSFORMER:
            inner = getattr(model, "transformer", model)
            return shard_transformer_specs(inner)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return sample inputs sized for the model's native i2v geometry.

        For the VAE, pass ``vae_type="decoder"`` (default) or ``"encoder"``.
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer_inputs(dtype)
        if self._variant == ModelVariant.VAE:
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return load_vae_decoder_inputs(dtype)
            if vae_type == "encoder":
                return load_vae_encoder_inputs(dtype)
            raise ValueError(f"Unknown vae_type: {vae_type}")
        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")
