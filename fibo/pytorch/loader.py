# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO (briaai/FIBO) model loader implementation.

FIBO is BRIA AI's 8B-parameter DiT-based flow-matching text-to-image model.
It uses SmolLM3-3B as the text encoder, Wan 2.2 as the VAE, and a novel
DimFusion conditioning architecture (paper: arXiv 2511.06876).

Rather than treating FIBO as a variant of some existing loader, this
introduces a dedicated ``fibo`` loader package. Layout mirrors
``stable_diffusion_3``, ``bria_2_3``, and ``hidream_i1`` — its preprocessing
tweaks, license-gated weight fetch (the FIBO repo on Hugging Face is gated
under the ``bria-fibo`` license), and bringup state can evolve independently
of any other model.

``load_model`` returns the FIBO transformer wrapped so it accepts the
positional tensor inputs the auto-runner expects. ``load_inputs`` returns
those positional tensors, captured by driving one short ``pipe(prompt=...)``
call and intercepting the first transformer forward — making the loader
robust to schema drift in upstream diffusers (``BriaFiboPipeline`` only lives
on diffusers git-main today).

Reference: https://huggingface.co/briaai/FIBO
"""

from typing import Optional

import torch
from diffusers import AutoencoderKLWan
from transformers import AutoModelForCausalLM

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
    BRINGUP_PROMPT,
    FiboTransformerWrapper,
    capture_transformer_inputs,
    load_pipe,
    positional_inputs_from_capture,
)
from .src.shard_specs import MESH_NAMES, build_shard_spec, get_mesh_shape


# FIBO's text tower is SmolLM3-3B, shipped in the ``text_encoder`` subfolder of
# the (license-gated) FIBO repo. It is a ``SmolLM3ForCausalLM`` used as an
# encoder: the pipeline reads its hidden states, never its LM head.
TEXT_ENCODER_SUBFOLDER = "text_encoder"
# SmolLM3-3B: vocab 128256, hidden 2048, 36 layers.
TEXT_ENCODER_VOCAB_SIZE = 128256
# The pipeline tokenizes with padding="longest" (up to max_sequence_length), so
# there is no fixed context length. 128 is a tile-aligned length that exercises
# the tower without inflating compile time.
TEXT_ENCODER_SEQ_LEN = 128
# Beginning-of-text id the pipeline forces for empty prompts.
BOT_TOKEN_ID = 128000


# FIBO's VAE is the Wan 2.2 video autoencoder (``AutoencoderKLWan``) used at a
# single frame. Latents are 5-D ``(B, z_dim, T, H, W)`` with T=1 for images, and
# the spatial compression is 16x — so a 512x512 image decodes from a 32x32 latent.
VAE_SUBFOLDER = "vae"
VAE_Z_DIM = 48
VAE_SPATIAL_SCALE = 16
VAE_DEFAULT_RESOLUTION = 512


class WanVaeDecoderWrapper(torch.nn.Module):
    """Run the Wan 2.2 VAE decode as a stateless module returning a plain tensor.

    ``BriaFiboPipeline`` decodes with
    ``self.vae.decode(latents_scaled, return_dict=False)[0]``, so ``decode`` —
    not the bare ``decoder`` submodule — is the unit the pipeline actually calls
    and the one worth comparing. ``return_dict=False`` keeps graph capture on a
    plain tensor rather than a ``DecoderOutput`` dataclass.

    Note this VAE is temporally causal and carries a ``feat_cache`` that
    ``_decode`` mutates per frame. At T=1 that loop runs exactly once, which is
    why the decode traces and lowers cleanly here.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents, return_dict=False)[0]


class SmolLM3TextEncoderWrapper(torch.nn.Module):
    """Run SmolLM3 as a stateless encoder returning a plain tensor.

    ``BriaFiboPipeline.get_prompt_embeds`` consumes the tower like this::

        encoder_outputs = self.text_encoder(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        hidden_states = encoder_outputs.hidden_states
        prompt_embeds = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)

    so the concatenation of the last two hidden states — not the LM logits — is
    the tensor the DiT is conditioned on, and the meaningful compilable unit.
    ``use_cache=False`` keeps the graph free of a KV cache (this is a single
    encoder pass, never autoregressive decode).
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        return torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)


class ModelVariant(StrEnum):
    """Available FIBO model variants."""

    BASE = "Base"
    TEXT_ENCODER = "TextEncoder"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """FIBO model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="briaai/FIBO",
        ),
        ModelVariant.TEXT_ENCODER: ModelConfig(
            pretrained_model_name="briaai/FIBO",
        ),
        ModelVariant.VAE: ModelConfig(
            pretrained_model_name="briaai/FIBO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Stub structured-JSON prompt used during input capture. FIBO is trained
    # on structured captions but the pipeline tokenizes via SmolLM3 either way.
    prompt = BRINGUP_PROMPT

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given FIBO variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self.pipe = None
        self._capture = None
        # FIBO inherits guidance_scale=5.0 from the model card's Generate example.
        self.guidance_scale = 5.0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIBO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=(
                ModelTask.NLP_EMBED_GEN
                if variant == ModelVariant.TEXT_ENCODER
                else ModelTask.CONDITIONAL_GENERATION
            ),
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load (and cache) the FIBO pipeline."""
        if self.pipe is None:
            self.pipe = load_pipe(
                self._variant_config.pretrained_model_name,
                dtype_override=dtype_override,
            )
        return self.pipe

    def _ensure_capture(self, dtype_override=None):
        """Capture transformer inputs once and cache the result."""
        if self._capture is not None:
            return self._capture
        self._load_pipeline(dtype_override=dtype_override)
        self._capture = capture_transformer_inputs(
            self.pipe,
            prompt=self.prompt,
            guidance_scale=self.guidance_scale,
        )
        return self._capture

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the wrapped FIBO transformer.

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the pipeline to.

        Returns:
            torch.nn.Module: ``FiboTransformerWrapper`` around the FIBO DiT,
            ready to accept the positional tensors returned by ``load_inputs``.
            For ``TEXT_ENCODER``, ``SmolLM3TextEncoderWrapper`` around the
            SmolLM3-3B text tower.
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            encoder = AutoModelForCausalLM.from_pretrained(
                self._variant_config.pretrained_model_name,
                subfolder=TEXT_ENCODER_SUBFOLDER,
                dtype=dtype_override if dtype_override is not None else torch.bfloat16,
            )
            return SmolLM3TextEncoderWrapper(encoder).eval()

        if self._variant == ModelVariant.VAE:
            vae = AutoencoderKLWan.from_pretrained(
                self._variant_config.pretrained_model_name,
                subfolder=VAE_SUBFOLDER,
                torch_dtype=(
                    dtype_override if dtype_override is not None else torch.bfloat16
                ),
            )
            return WanVaeDecoderWrapper(vae).eval()

        self._ensure_capture(dtype_override=dtype_override)
        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)
        return FiboTransformerWrapper(self.pipe.transformer, self._capture)

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return positional tensor inputs for the FIBO transformer.

        Args:
            dtype_override: Optional ``torch.dtype`` for tensor inputs.
                Non-tensor inputs (e.g. ``joint_attention_kwargs``) are passed
                through unchanged.
            batch_size: Ignored for now — the pipeline call always produces
                ``batch_size=1`` * CFG (2 effective). Retained for the
                signature the auto-runner expects.

        Returns:
            tuple: Positional inputs matching ``FiboTransformerWrapper.forward``.
            For ``TEXT_ENCODER``, ``(input_ids, attention_mask)``.
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            # int64 token ids — dtype_override applies to floating-point inputs
            # only, and this tower takes none.
            generator = torch.Generator().manual_seed(0)
            input_ids = torch.randint(
                0,
                TEXT_ENCODER_VOCAB_SIZE,
                (batch_size, TEXT_ENCODER_SEQ_LEN),
                dtype=torch.long,
                generator=generator,
            )
            # The pipeline always tokenizes with add_special_tokens=True, so the
            # sequence opens with the beginning-of-text id.
            input_ids[:, 0] = BOT_TOKEN_ID
            return input_ids, torch.ones_like(input_ids)

        if self._variant == ModelVariant.VAE:
            # 5-D latent (B, z_dim, T, H, W); T=1 for images, spatial scale 16.
            latent_hw = VAE_DEFAULT_RESOLUTION // VAE_SPATIAL_SCALE
            generator = torch.Generator().manual_seed(0)
            latents = torch.randn(
                batch_size,
                VAE_Z_DIM,
                1,
                latent_hw,
                latent_hw,
                generator=generator,
            )
            return (
                latents.to(dtype_override)
                if dtype_override is not None
                else latents.to(torch.bfloat16),
            )

        capture = self._ensure_capture(dtype_override=dtype_override)
        inputs = positional_inputs_from_capture(capture)

        if dtype_override is None:
            return inputs

        cast = []
        for value in inputs:
            if torch.is_tensor(value) and value.is_floating_point():
                cast.append(value.to(dtype_override))
            else:
                cast.append(value)
        return tuple(cast)

    def get_mesh_config(self, num_devices: int):
        """Return ``(mesh_shape, mesh_names)`` for tensor-parallel execution.

        FIBO is an 8B DiT that runs out of DRAM on a single chip. It is brought
        up across multiple chips with Megatron-1D tensor parallelism over a
        ``(None, "model")`` mesh. See ``src/shard_specs.py``.

        Args:
            num_devices: Total chip count (``xr.global_runtime_device_count()``).

        Returns:
            tuple: ``(mesh_shape, mesh_names)`` consumed by the auto-runner.
        """
        if self._variant in (ModelVariant.TEXT_ENCODER, ModelVariant.VAE):
            # SmolLM3-3B and the 705M Wan VAE each fit on a single chip, so both
            # component towers are brought up
            # single-device and unsharded. Tensor-parallel shard specs for it are
            # a follow-up: they cannot be validated on one chip, where the mesh
            # collapses to (1, 1) and any spec is a no-op.
            return (1, 1), MESH_NAMES
        return get_mesh_shape(num_devices)

    def load_shard_spec(self, model):
        """Return the tensor -> partition-spec mapping for the FIBO transformer.

        Args:
            model: the ``FiboTransformerWrapper`` returned by ``load_model``.

        Returns:
            dict: ``{torch.nn.Parameter: partition_spec}``. Parameters absent
            from the mapping are replicated across the mesh. ``None`` for the
            text encoder, which is unsharded.
        """
        if self._variant in (ModelVariant.TEXT_ENCODER, ModelVariant.VAE):
            return None
        return build_shard_spec(model)
