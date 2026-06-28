# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO model loader implementation for text-to-image generation.

SRPO (tencent/SRPO) ships only the denoiser weights -- a FLUX.1-dev
``FluxTransformer2DModel`` fine-tuned with Semantic Relative Preference
Optimization (arXiv:2509.06942). The repo contains a single
``diffusion_pytorch_model.safetensors`` (diffusers key layout, no config), so
the loader builds the transformer architecture from the FLUX.1-dev transformer
config and injects the SRPO state dict. The remaining pipeline components
(CLIP + T5 text encoders, VAE, scheduler, tokenizers) are reused from
FLUX.1-dev.
"""
import types
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from typing import Optional


def _single_block_forward_split(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    """Drop-in for ``FluxSingleTransformerBlock.forward`` that replaces the
    fused ``cat([attn, mlp], dim=2) -> proj_out`` with two equivalent matmuls.

    The original concat builds a (dim + mlp_hidden)=15360-wide activation whose
    per-core L1 circular-buffer page (~1.5 MB) exceeds Blackhole L1 (~1.4 MB) at
    native 1024x1024. Splitting ``proj_out`` along its input into the attention
    and MLP halves is mathematically identical (concat-then-matmul == sum of the
    two slice matmuls) and removes the oversized concat op.
    """
    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )
    gate = gate.unsqueeze(1)
    proj = self.proj_out_attn(attn_output) + self.proj_out_mlp(mlp_hidden_states)
    hidden_states = gate * proj
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    encoder_hidden_states, hidden_states = (
        hidden_states[:, :text_seq_len],
        hidden_states[:, text_seq_len:],
    )
    return encoder_hidden_states, hidden_states

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

    SRPO = "srpo"


class ModelLoader(ForgeModel):
    """SRPO model loader implementation for text-to-image generation tasks."""

    # The denoiser ships here; everything else comes from FLUX.1-dev.
    _BASE_PIPELINE = "black-forest-labs/FLUX.1-dev"
    _SRPO_WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.SRPO: ModelConfig(
            pretrained_model_name="tencent/SRPO",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SRPO

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipe = None
        # FLUX.1-dev is a guidance-distilled model; SRPO keeps the same
        # guidance-embedding interface. Default guidance_scale matches FLUX.1-dev.
        self.guidance_scale = 3.5

    @staticmethod
    def apply_single_block_proj_out_split(model):
        """Apply the FLUX single-block ``proj_out`` split TT workaround in place.

        Opt-in (not applied by ``load_model``) so the loader stays a faithful
        FLUX denoiser for CPU/golden use; the device path calls this to avoid
        the 15360-wide concat that overflows Blackhole L1 at native resolution.
        After this, ``load_shard_spec`` shards ``proj_out_attn``/``proj_out_mlp``
        instead of ``proj_out``.
        """
        for block in model.single_transformer_blocks:
            if hasattr(block, "proj_out_attn"):
                continue
            proj = block.proj_out
            dim = proj.out_features
            attn_lin = nn.Linear(dim, dim, bias=proj.bias is not None)
            mlp_lin = nn.Linear(proj.in_features - dim, dim, bias=False)
            with torch.no_grad():
                attn_lin.weight.copy_(proj.weight[:, :dim])
                mlp_lin.weight.copy_(proj.weight[:, dim:])
                if proj.bias is not None:
                    attn_lin.bias.copy_(proj.bias)
            attn_lin = attn_lin.to(proj.weight.dtype)
            mlp_lin = mlp_lin.to(proj.weight.dtype)
            block.proj_out_attn = attn_lin
            block.proj_out_mlp = mlp_lin
            del block.proj_out
            block.forward = types.MethodType(_single_block_forward_split, block)
        return model

    def get_mesh_config(self, num_devices: int):
        """Return ``(mesh_shape, mesh_names)`` for a ("batch", "model") 2D mesh.

        SRPO/FLUX is sharded tensor-parallel on the model axis only, so the
        batch axis is 1 and the model axis spans all devices. The 11.9B bf16
        denoiser (~24 GB) does not fit a single 32 GB Blackhole chip at native
        1024x1024 (a single activation concat needs ~4.8 GB on top of the
        weights), so multi-chip TP is required, not merely a perf baseline.
        """
        return (1, num_devices), ("batch", "model")

    def load_shard_spec(self, model):
        """Return ``{param_tensor: partition_spec}`` (Megatron column->row TP).

        ``model`` is the FLUX ``FluxTransformer2DModel`` (the SRPO denoiser).
        Attention q/k/v and the MLP up-projections are column-parallel
        (``("model", None)``); attention/MLP output projections are
        row-parallel (``(None, "model")``). The AdaLN modulation linears are
        **row-parallel** (shard the temb input, all-reduce the modulation):
        replicating or column-sharding them makes a downstream activation
        concat overflow Blackhole per-core L1 at native 1024; row-sharding the
        modulation keeps that concat within L1. Embedders and the final
        projection stay replicated. hidden_size=3072 and 24 heads are both
        divisible by the supported device counts (1, 2, 4, 8).
        """
        specs = {}

        def col(linear):
            specs[linear.weight] = ("model", None)
            if getattr(linear, "bias", None) is not None:
                specs[linear.bias] = ("model",)

        def row(linear):
            specs[linear.weight] = (None, "model")
            # row-parallel bias is added after the all-reduce -> keep replicated

        # Double-stream blocks: dual image/text attention + two FFNs.
        for block in model.transformer_blocks:
            for attn_in in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj",
                            "add_v_proj"):
                col(getattr(block.attn, attn_in))
            row(block.attn.to_out[0])
            row(block.attn.to_add_out)
            col(block.ff.net[0].proj)
            row(block.ff.net[2])
            col(block.ff_context.net[0].proj)
            row(block.ff_context.net[2])
            # AdaLN modulation -> row-parallel (L1 concat fix)
            row(block.norm1.linear)
            row(block.norm1_context.linear)

        # Single-stream blocks: fused attention + MLP, joint output projection.
        for block in model.single_transformer_blocks:
            for attn_in in ("to_q", "to_k", "to_v"):
                col(getattr(block.attn, attn_in))
            col(block.proj_mlp)  # MLP up-projection (column-parallel)
            if hasattr(block, "proj_out_attn"):
                # proj_out split TT workaround applied -> shard both halves
                row(block.proj_out_attn)
                row(block.proj_out_mlp)
            else:
                row(block.proj_out)  # joint [attn|mlp] output projection
            row(block.norm.linear)  # AdaLN modulation -> row-parallel

        # Final AdaLN-continuous modulation -> row-parallel (L1 concat fix)
        row(model.norm_out.linear)

        return specs

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SRPO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,  # text-to-image
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_transformer(self, dtype_override=None):
        """Build the FLUX transformer architecture and inject the SRPO weights.

        SRPO publishes only the denoiser state dict (diffusers key layout) with
        no config.json, so the architecture is instantiated from the FLUX.1-dev
        transformer config and the SRPO state dict is loaded into it.
        """
        from diffusers import FluxTransformer2DModel

        config = FluxTransformer2DModel.load_config(
            self._BASE_PIPELINE, subfolder="transformer"
        )
        transformer = FluxTransformer2DModel.from_config(config)

        weights_path = hf_hub_download(
            self._variant_config.pretrained_model_name, self._SRPO_WEIGHTS_FILE
        )
        state_dict = load_file(weights_path)
        transformer.load_state_dict(state_dict, strict=True)

        if dtype_override is not None:
            transformer = transformer.to(dtype_override)
        return transformer

    def _load_pipeline(self, dtype_override=None):
        """Load the FLUX.1-dev pipeline with the SRPO transformer swapped in.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            The loaded pipeline instance
        """
        from diffusers import FluxPipeline

        transformer = self._build_transformer(dtype_override=dtype_override)

        pipe_kwargs = {"use_safetensors": True, "transformer": transformer}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = FluxPipeline.from_pretrained(self._BASE_PIPELINE, **pipe_kwargs)
        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SRPO transformer (denoiser) for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its default dtype.

        Returns:
            torch.nn.Module: The SRPO FLUX transformer (denoiser) for text-to-image.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1, height=1024, width=1024):
        """Build sample inputs for the SRPO transformer at native resolution.

        Mirrors the FLUX text-to-image input preparation: CLIP-pooled and T5
        prompt embeddings, packed latents, and RoPE position ids. Defaults to
        the FLUX.1-dev native 1024x1024 resolution.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Optional batch size override (default 1).
            height: Output image height in pixels (default 1024, native).
            width: Output image width in pixels (default 1024, native).

        Returns:
            dict: Input tensors for the transformer forward.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = 512  # FLUX.1-dev native T5 sequence length
        prompt = "An astronaut riding a horse in a futuristic city"
        do_classifier_free_guidance = self.guidance_scale > 1.0
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        # CLIP text encoding (pooled)
        text_inputs_clip = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.pipe.text_encoder(
            text_inputs_clip.input_ids, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        # T5 text encoding (sequence)
        text_inputs_t5 = self.pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.pipe.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Packed latents at native resolution
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))

        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
        )
        latents = torch.randn(shape, dtype=dtype)
        latents = latents.view(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent // 2,
            2,
            width_latent // 2,
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size * num_images_per_prompt,
            (height_latent // 2) * (width_latent // 2),
            num_channels_latents * 4,
        )

        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        if do_classifier_free_guidance:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

        inputs = {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
