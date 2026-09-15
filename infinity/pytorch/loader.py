# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Infinity component loader.

Each variant corresponds to one independently loadable component:
  - TextEncoder → T5-XL encoder (T5TextEncoderWrapper)
  - Transformer → Infinity 2B transformer
  - Vae         → BSQ-VAE decoder (VAEDecoderWrapper, decoder-only)

All three share the ``(1, num_devices)`` / ``(None, "model")`` mesh, so the
pipeline can place every component on one mesh.
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
    PN,
    REPO_ID,
    TRANSFORMER_NUM_HEADS,
    VAEDecoderWrapper,
    build_forward_inputs,
    build_run_args,
    load_text_encoder,
    load_text_encoder_inputs,
    load_tokenizer,
    load_tokenizer_and_encoder,
    load_transformer,
    load_vae,
    load_vae_inputs,
    shard_text_encoder_specs,
    shard_transformer_specs,
    shard_vae_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the Infinity pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual Infinity components without pulling the full pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=REPO_ID),
    }

    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    _PN = PN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        # Side effects of load_model(), kept so load_inputs() can build
        # realistic tensors and so the pipeline can reach the pieces that are
        # not the returned module (tokenizer, the VAE's quantizer).
        self.tokenizer = None
        self.vae = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="Infinity",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def _build_run_args(self):
        """SimpleNamespace mirroring the args ``run_infinity``'s loaders read."""
        return build_run_args()

    def load_model(
        self, *, dtype_override: Optional[torch.dtype] = None, vae=None, **kwargs
    ):
        """Load and return the component for this variant as a ``torch.nn.Module``.

        Returns:
            TEXT_ENCODER → T5TextEncoderWrapper (bare last_hidden_state)
            TRANSFORMER  → Infinity transformer
            VAE          → VAEDecoderWrapper (decoder-only, returns a plain tensor)

        Args:
            dtype_override: weight dtype; defaults to bf16 for TT execution.
            vae: optional pre-loaded BSQ-VAE, honoured by TRANSFORMER and VAE so a
                caller holding one (the pipeline) does not load it twice.
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            self.tokenizer = load_tokenizer()
            self.model = load_text_encoder(dtype)
            return self.model

        if self._variant == ModelVariant.VAE:
            self.vae = vae if vae is not None else load_vae()
            self.model = VAEDecoderWrapper(self.vae, dtype)
            return self.model

        if self._variant == ModelVariant.TRANSFORMER:
            # The transformer reads embed_dim / vocab_size / the bit-label mask
            # off the VAE, and load_inputs needs the text encoder as well.
            self.vae = vae if vae is not None else load_vae()
            self.tokenizer = load_tokenizer()
            self.model = load_transformer(self.vae, dtype)
            return self.model

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of positional input tensors for the active component.

        TEXT_ENCODER → [input_ids (1,512) int64, attention_mask (1,512) int64]
        TRANSFORMER  → [label_B_or_BLT, x_BLC_wo_prefix, scale_schedule], the
                       training-style ``forward`` path (``cfg_infer=False``) -- a
                       single traceable pass returning logits, not a sampling
                       loop. A positional sequence (not a dict) is required
                       because the test infra invokes the model as
                       ``model(*inputs)``.
        VAE          → [z (1, 32, 64, 64)]

        Args:
            dtype_override: optional dtype for the float tensors.
            batch_size: TRANSFORMER only -- prompt replication factor.
            prompt: TEXT_ENCODER / TRANSFORMER only -- prompt override.
        """
        if self._variant == ModelVariant.VAE:
            return load_vae_inputs(dtype_override)

        if self.tokenizer is None:
            raise RuntimeError("load_model() must be called before load_inputs().")

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder_inputs(
                self.tokenizer, dtype_override, kwargs.get("prompt")
            )

        if self._variant == ModelVariant.TRANSFORMER:
            # The conditioning tensors need a text encoder to produce them; the
            # transformer variant does not keep one placed, so use the reference
            # pair here.
            _, text_encoder = load_tokenizer_and_encoder()
            forward_inputs = build_forward_inputs(
                tokenizer=self.tokenizer,
                text_encoder=text_encoder,
                vae=self.vae,
                # A smaller preset ("0.06M" -> 7 scales, L=521) keeps the O(L^2)
                # attention small enough to run replicated, which is what makes a
                # sharded-vs-unsharded comparison possible at all; the weights and
                # the rope grid are schedule-independent.
                pn=kwargs.get("pn", self._PN),
                batch_size=kwargs.get("batch_size", 1),
                prompt=kwargs.get("prompt"),
                dtype_override=dtype_override,
            )
            return [
                forward_inputs["label_B_or_BLT"],
                forward_inputs["x_BLC_wo_prefix"],
                forward_inputs["scale_schedule"],
            ]

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return ``(mesh_shape, axis_names)`` for the shared component mesh.

        The mochi-style 2D mesh ``(1, num_devices)`` with axis names
        ``(None, "model")``: the ``model`` axis (index 1) carries the tensor
        parallelism and the size-1 axis is named ``None`` so no partition spec
        ever references it. Inputs are batch-1, so there is no data-parallel
        axis. For the transformer the ``model`` axis splits the 16 attention
        heads evenly (4-per-device on 4 devices, 2-per-device on 8), shrinking
        the O(L^2) self-attention score tensor from ``[1, 16, L, L]`` to
        ``[1, 16 // num_devices, L, L]`` -- the buffer that OOM'd when attention
        was replicated.

        Args:
            num_devices: Total devices visible to the runtime.

        Returns:
            tuple: ``(mesh_shape, axis_names)``.
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Infinity sharding currently supports "
                f"{sorted(MESH_SHAPES)} devices, got {num_devices}."
            )
        if TRANSFORMER_NUM_HEADS % num_devices != 0:
            raise ValueError(
                f"Infinity head-parallel sharding needs num_devices to divide "
                f"the {TRANSFORMER_NUM_HEADS} attention heads, got {num_devices}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return a ``tensor -> partition_spec`` dict for the active component.

        Expects the same model object ``load_model()`` returned:
          TEXT_ENCODER → T5TextEncoderWrapper (q/k/v column, o row; wi column,
                         wo row -- two all-reduces per block)
          TRANSFORMER  → Infinity transformer (head-parallel attention + FFN,
                         one all-reduce per attention/FFN pair)
          VAE          → VAEDecoderWrapper (each ResnetBlock's conv1 column /
                         conv2 row -- one all-reduce per block, 17 in total)
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model)
        if self._variant == ModelVariant.VAE:
            return shard_vae_specs(model)
        return None
