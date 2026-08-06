# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO (tencent/SRPO) model loader implementation.

SRPO is a FLUX.1-dev fine-tune from Tencent Hunyuan that publishes only the
transformer weights (``diffusion_pytorch_model.safetensors``). Rather than
adding it as a variant of the existing ``flux`` loader, this introduces a
dedicated loader package so its preprocessing tweaks, license-gated weights,
and bringup state can evolve independently. This mirrors the layout used by
``stable_diffusion_3`` and ``bria_2_3``.

``load_model`` returns the FLUX transformer (with SRPO weights overlaid) as
an ``nn.Module``. ``load_inputs`` returns the positional tensors the FLUX
transformer consumes — the same shape contract as ``flux/pytorch/loader.py``.

Reference: https://huggingface.co/tencent/SRPO
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
from .src.model_utils import load_pipe, srpo_preprocessing
from .src.shard_specs import build_shard_spec, get_mesh_shape


class ModelVariant(StrEnum):
    """Available SRPO model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """SRPO model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="tencent/SRPO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Prompt taken from the SRPO Hugging Face model card.
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given SRPO variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self.pipe = None
        # SRPO inherits FLUX.1-dev's guidance scale (3.5 per the model card).
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SRPO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load (and cache) the SRPO pipeline (FLUX.1-dev base + SRPO weights)."""
        self.pipe = load_pipe(
            self._variant_config.pretrained_model_name,
            dtype_override=dtype_override,
        )
        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the SRPO transformer (FLUX.1-dev architecture, SRPO weights).

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the pipeline to.

        Returns:
            torch.nn.Module: The FLUX transformer with SRPO weights overlaid.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)
        elif dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return positional inputs for the FLUX transformer (SRPO weights).

        Args:
            dtype_override: Optional ``torch.dtype`` for the returned tensors.
            batch_size: Batch size for the synthetic input. Defaults to 1.

        Returns:
            dict: Input tensors that can be fed directly to the transformer
            (matches the keyword-argument signature of FLUX's
            ``FluxTransformer2DModel.forward``).
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        (
            hidden_states,
            timestep,
            guidance,
            pooled_projections,
            encoder_hidden_states,
            txt_ids,
            img_ids,
        ) = srpo_preprocessing(
            self.pipe,
            self.prompt,
            dtype=dtype,
            batch_size=batch_size,
            guidance_scale=self.guidance_scale,
        )

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

    def get_mesh_config(self, num_devices: int):
        """Return ``(mesh_shape, mesh_names)`` for tensor-parallel execution.

        SRPO is a ~12B FLUX DiT that runs out of DRAM on a single chip; it is
        brought up across multiple chips with Megatron-1D tensor parallelism
        over a ``(None, "model")`` mesh. See ``src/shard_specs.py``.

        Args:
            num_devices: Total chip count (``xr.global_runtime_device_count()``).

        Returns:
            tuple: ``(mesh_shape, mesh_names)`` consumed by the auto-runner.
        """
        return get_mesh_shape(num_devices)

    def load_shard_spec(self, model):
        """Return the tensor -> partition-spec mapping for the SRPO transformer.

        Args:
            model: the ``FluxTransformer2DModel`` returned by ``load_model``.

        Returns:
            dict: ``{torch.nn.Parameter: partition_spec}``. Parameters absent
            from the mapping are replicated across the mesh.
        """
        return build_shard_spec(model)

    # ------------------------------------------------------------------ #
    # TAEF1 lightweight preview VAE decoder on TT.
    #
    # SRPO's full AutoencoderKL VAE noises out / OOMs on TT at native res.
    # TAEF1 (madebyollin/taef1) is the tiny FLUX autoencoder — its conv-only
    # decoder (no complex/FFT, no GroupNorm) is the tractable preview decoder
    # that runs on TT and decodes SRPO's FLUX-native latents. See tt-xla #5537.
    # ------------------------------------------------------------------ #

    TAEF1_REPO = "madebyollin/taef1"

    def load_taef1_decoder(self):
        """Load the TAEF1 tiny FLUX autoencoder (preview VAE)."""
        from diffusers import AutoencoderTiny

        self.taef1 = (
            AutoencoderTiny.from_pretrained(self.TAEF1_REPO, torch_dtype=torch.float32)
            .eval()
        )
        return self.taef1

    def decode_taef1(self, latents, on_tt=False):
        """Decode FLUX latents [B, 16, H/8, W/8] -> image [-1, 1] via TAEF1.

        With ``on_tt=True`` the conv decoder runs on TT via
        ``torch.compile(backend="tt")``. ``AutoencoderTiny.decode(z)`` is
        ``self.decoder(z)`` directly (no pre-scaling), so the raw latents feed
        the decoder in both modes. ``load_taef1_decoder`` must run first.
        """
        vae = getattr(self, "taef1", None) or self.load_taef1_decoder()
        with torch.no_grad():
            if not on_tt:
                return vae.decode(latents).sample

            import torch_xla  # noqa: F401
            import torch_xla.core.xla_model as xm

            dev = xm.xla_device()
            dec = vae.decoder.to(dtype=torch.bfloat16).to(dev)
            compiled = torch.compile(lambda z: dec(z), backend="tt")
            out = compiled(latents.to(dtype=torch.bfloat16).to(dev))
            return out.to("cpu").to(torch.float32)
