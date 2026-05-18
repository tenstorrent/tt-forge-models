# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HiDream-I1-Fast component loader.

Each variant corresponds to one independently loadable component:
  - TextEncoder    → CLIPTextModelWithProjection (CLIP ViT-L/14)         params=0.123 B
  - TextEncoder2   → CLIPTextModelWithProjection (OpenCLIP ViT-bigG/14)  params=0.695 B
  - TextEncoder3   → T5EncoderModel (T5 v1.1 XXL encoder)                params=4.6 B
  - TextEncoder4   → LlamaForCausalLM (Llama-3.1-8B-Instruct)            params=8.0 B
  - Transformer    → HiDreamImageTransformer2DModel (Sparse-MoE MM-DiT)  params=17 B
  - Vae            → AutoencoderKL (FLUX-derived 16-channel latent)      params=0.084 B

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
    CLIP_VOCAB_SIZE,
    DTYPE,
    HIDREAM_REPO_ID,
    LATENT_CHANNELS,
    LATENT_H,
    LATENT_W,
    LLAMA_HIDDEN,
    LLAMA_NUM_LAYERS,
    LLAMA_REPO_ID,
    LLAMA_VOCAB_SIZE,
    MAX_SEQ_LEN,
    MESH_NAMES,
    MESH_SHAPES,
    POOLED_TEXT_EMB_DIM,
    T5_HIDDEN,
    T5_VOCAB_SIZE,
    CLIPPooledWrapper,
    HiDreamTransformerWrapper,
    LlamaStackedHiddenWrapper,
    T5EncoderWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_text_encoder_2,
    load_text_encoder_3,
    load_text_encoder_4,
    load_transformer,
    load_vae,
    shard_hidream_transformer_specs,
    shard_llama_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the HiDream-I1-Fast pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TEXT_ENCODER_2 = "TextEncoder2"
    TEXT_ENCODER_3 = "TextEncoder3"
    TEXT_ENCODER_4 = "TextEncoder4"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual HiDream-I1-Fast components without pulling the full pipeline.

    text_encoder, text_encoder_2, text_encoder_3, transformer, and vae weights
    come from HiDream-ai/HiDream-I1-Fast. text_encoder_4 (Llama-3.1-8B) is loaded
    separately from the Meta repo (HiDream snapshot does not ship Llama).
    """

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=HIDREAM_REPO_ID),
        ModelVariant.TEXT_ENCODER_2: ModelConfig(pretrained_model_name=HIDREAM_REPO_ID),
        ModelVariant.TEXT_ENCODER_3: ModelConfig(pretrained_model_name=HIDREAM_REPO_ID),
        ModelVariant.TEXT_ENCODER_4: ModelConfig(pretrained_model_name=LLAMA_REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=HIDREAM_REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=HIDREAM_REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        text_variants = (
            ModelVariant.TEXT_ENCODER,
            ModelVariant.TEXT_ENCODER_2,
            ModelVariant.TEXT_ENCODER_3,
            ModelVariant.TEXT_ENCODER_4,
        )
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in text_variants
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="HiDreamI1Fast",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module."""
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return CLIPPooledWrapper(load_text_encoder(model_name, dtype))
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return CLIPPooledWrapper(load_text_encoder_2(model_name, dtype))
        if self._variant == ModelVariant.TEXT_ENCODER_3:
            return T5EncoderWrapper(load_text_encoder_3(model_name, dtype))
        if self._variant == ModelVariant.TEXT_ENCODER_4:
            return LlamaStackedHiddenWrapper(load_text_encoder_4(model_name, dtype))
        if self._variant == ModelVariant.TRANSFORMER:
            return HiDreamTransformerWrapper(load_transformer(model_name, dtype))
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(model_name, dtype))

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Sharded variants: TEXT_ENCODER_4 and TRANSFORMER. The other four fit on
        a single chip and always map to (1, 1).

        Supported device counts for sharded variants: 1, 2, 4, 8, 32.
        """
        sharded_variants = (ModelVariant.TEXT_ENCODER_4, ModelVariant.TRANSFORMER)
        if self._variant not in sharded_variants:
            return (1, 1), MESH_NAMES

        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor → partition_spec dict for the active component.

        Expects the same model object returned by load_model():
          TEXT_ENCODER_4 → LlamaStackedHiddenWrapper (specs built from .llama)
          TRANSFORMER    → HiDreamTransformerWrapper (specs built from .transformer)
          Others         → None (single-chip, no sharding)
        """
        if self._variant == ModelVariant.TEXT_ENCODER_4:
            return shard_llama_specs(model.llama)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_hidream_transformer_specs(model.transformer)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER    → [input_ids (1,128) int64]
        TEXT_ENCODER_2  → [input_ids (1,128) int64]
        TEXT_ENCODER_3  → [input_ids (1,128) int64, attention_mask (1,128) int64]
        TEXT_ENCODER_4  → [input_ids (1,128) int64, attention_mask (1,128) int64]
        TRANSFORMER     → [hidden_states (1,16,128,128), timesteps (1,), enc_t5 (1,128,4096),
                           enc_llama3 (32,1,128,4096), pooled_embeds (1,2048)]
        VAE             → [z (1,16,128,128)]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, CLIP_VOCAB_SIZE, (1, MAX_SEQ_LEN), dtype=torch.long
            )
            return [input_ids]

        if self._variant == ModelVariant.TEXT_ENCODER_2:
            input_ids = torch.randint(
                0, CLIP_VOCAB_SIZE, (1, MAX_SEQ_LEN), dtype=torch.long
            )
            return [input_ids]

        if self._variant == ModelVariant.TEXT_ENCODER_3:
            input_ids = torch.randint(
                0, T5_VOCAB_SIZE, (1, MAX_SEQ_LEN), dtype=torch.long
            )
            attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TEXT_ENCODER_4:
            input_ids = torch.randint(
                0, LLAMA_VOCAB_SIZE, (1, MAX_SEQ_LEN), dtype=torch.long
            )
            attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = torch.randn(
                1, LATENT_CHANNELS, LATENT_H, LATENT_W, dtype=dtype
            )
            timesteps = torch.tensor([1.0], dtype=torch.float32)
            encoder_hidden_states_t5 = torch.randn(
                1, MAX_SEQ_LEN, T5_HIDDEN, dtype=dtype
            )
            encoder_hidden_states_llama3 = torch.randn(
                LLAMA_NUM_LAYERS, 1, MAX_SEQ_LEN, LLAMA_HIDDEN, dtype=dtype
            )
            pooled_embeds = torch.randn(1, POOLED_TEXT_EMB_DIM, dtype=dtype)
            return [
                hidden_states,
                timesteps,
                encoder_hidden_states_t5,
                encoder_hidden_states_llama3,
                pooled_embeds,
            ]

        if self._variant == ModelVariant.VAE:
            z = torch.randn(1, LATENT_CHANNELS, LATENT_H, LATENT_W, dtype=dtype)
            return [z]

        raise ValueError(f"Unknown variant: {self._variant}")
