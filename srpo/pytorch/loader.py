# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO model loader implementation for text-to-image generation.

tencent/SRPO is a FLUX.1-dev fine-tune that ships ONLY the MMDiT transformer
(``diffusion_pytorch_model.safetensors``). The VAE, text encoders (CLIP + T5),
tokenizers and scheduler are reused from ``black-forest-labs/FLUX.1-dev`` (gated
- requires an HF token). The transformer architecture is identical to FLUX.1-dev,
so we build it ``from_config`` (no base-transformer download) and inject the SRPO
state dict.

The denoiser (transformer) is the key, heavy component validated on device. The
remaining components (CLIP, T5, VAE) are standard FLUX components and stay in host
Python during the composite pipeline.
"""
from typing import Optional

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
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

# Base FLUX.1-dev repo supplies the transformer config and the non-transformer
# pipeline components (VAE, CLIP, T5, tokenizers, scheduler).
_FLUX_BASE = "black-forest-labs/FLUX.1-dev"
_SRPO_REPO = "tencent/SRPO"
_SRPO_WEIGHTS = "diffusion_pytorch_model.safetensors"


class ModelVariant(StrEnum):
    """Available SRPO model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """SRPO model loader implementation for text-to-image generation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name=_SRPO_REPO,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Native generation resolution (model card / FLUX.1-dev default)
    SAMPLE_SIZE = 1024
    # FlowMatch steps recommended by the SRPO model card
    NUM_INFERENCE_STEPS = 50
    GUIDANCE_SCALE = 3.5
    # T5 sequence length used by FLUX.1-dev
    MAX_SEQUENCE_LENGTH = 512

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipe = None
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
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

    def _load_transformer(self, dtype_override=None):
        """Build the FLUX transformer from config and inject SRPO weights.

        Builds the architecture from the FLUX.1-dev transformer config (no
        base-transformer download) and loads the SRPO state dict strictly.
        """
        if self._transformer is not None:
            return self._transformer

        config = FluxTransformer2DModel.load_config(
            _FLUX_BASE, subfolder="transformer"
        )
        transformer = FluxTransformer2DModel.from_config(config)

        weights_path = hf_hub_download(_SRPO_REPO, _SRPO_WEIGHTS)
        state_dict = load_file(weights_path)
        transformer.load_state_dict(state_dict, strict=True)

        if dtype_override is not None:
            transformer = transformer.to(dtype_override)

        transformer.eval()
        self._transformer = transformer
        return transformer

    def _load_pipeline(self, dtype_override=None):
        """Load the full FluxPipeline with the SRPO transformer injected.

        Reuses every non-transformer component (VAE, CLIP, T5, tokenizers,
        scheduler) from FLUX.1-dev and swaps in the SRPO transformer. Used by the
        composite generation path; the per-component device gate only needs
        ``load_model`` (the transformer).
        """
        if self.pipe is not None:
            return self.pipe

        transformer = self._load_transformer(dtype_override=dtype_override)

        pipe_kwargs = {"transformer": transformer}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = FluxPipeline.from_pretrained(_FLUX_BASE, **pipe_kwargs)
        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SRPO transformer (denoiser) for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype (the SRPO weights ship as fp32).

        Returns:
            torch.nn.Module: The FLUX/SRPO transformer (denoiser).
        """
        return self._load_transformer(dtype_override=dtype_override)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return sample single-step denoiser inputs at native 1024x1024.

        Inputs are synthesized at the correct native shapes (no text-encoder
        download required for the denoiser gate). Latent token count and image
        IDs are derived from the 1024x1024 native resolution.

        Args:
            dtype_override: Optional torch.dtype for the input tensors.
            batch_size: Batch size (default 1).

        Returns:
            dict: Input tensors for the transformer forward pass.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # FLUX latent geometry for the native resolution.
        vae_scale_factor = 8
        height = width = self.SAMPLE_SIZE
        height_latent = 2 * (height // (vae_scale_factor * 2))  # 128
        width_latent = 2 * (width // (vae_scale_factor * 2))  # 128
        num_channels_latents = 16  # transformer.in_channels // 4 (64 // 4)

        # Packed latent tokens: [B, (H/2)*(W/2), C*4]
        h = height_latent // 2  # 64
        w = width_latent // 2  # 64
        seq_latent = h * w  # 4096
        latents = torch.randn(
            batch_size, seq_latent, num_channels_latents * 4, dtype=dtype
        )

        # Latent image position IDs [seq_latent, 3]
        latent_image_ids = torch.zeros(h, w, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(h)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(w)[None, :]
        img_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # T5 conditioning (synthetic, correct shapes)
        seq_t5 = self.MAX_SEQUENCE_LENGTH
        prompt_embeds = torch.randn(batch_size, seq_t5, 4096, dtype=dtype)
        txt_ids = torch.zeros(seq_t5, 3, dtype=dtype)

        # CLIP pooled projection (synthetic)
        pooled_prompt_embeds = torch.randn(batch_size, 768, dtype=dtype)

        guidance = torch.full([batch_size], self.GUIDANCE_SCALE, dtype=dtype)

        inputs = {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0] * batch_size, dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            "joint_attention_kwargs": {},
        }
        return inputs

    # ------------------------------------------------------------------
    # Tensor-parallel hooks (perf starting point only).
    #
    # The single 32 GB Blackhole chip fits the bf16 denoiser at native 1024 via
    # torch.compile, so TP is NOT required for the bringup baseline. A raw-xla TP
    # path (Megatron column->row, mesh (1,N)) was attempted and cleared OOM/L1 but
    # diverged numerically; single-chip torch.compile is the adopted baseline.
    # These hooks are shipped as a perf-tuning starting point.
    # ------------------------------------------------------------------
    def get_mesh_config(self, num_devices: int):
        """Megatron-style mesh: (batch=1, model=num_devices)."""
        mesh_shape = (1, num_devices)
        mesh_names = ("batch", "model")
        return mesh_shape, mesh_names

    def load_shard_spec(self, model):
        """Megatron column->row shard plan for the FLUX/SRPO MMDiT blocks.

        Column-shard the attention/MLP input projections on the model axis and
        row-shard the output projections. Only emitted for params whose shardable
        dim divides the model axis.
        """
        shard_spec = {}
        for name, param in model.named_parameters():
            # Attention input projections (column-shard)
            if name.endswith(("to_q.weight", "to_k.weight", "to_v.weight")) or (
                "add_q_proj" in name
                or "add_k_proj" in name
                or "add_v_proj" in name
            ):
                shard_spec[param] = ("model", None)
            # Attention output projections (row-shard)
            elif name.endswith(("to_out.0.weight", "to_add_out.weight")):
                shard_spec[param] = (None, "model")
        return shard_spec
