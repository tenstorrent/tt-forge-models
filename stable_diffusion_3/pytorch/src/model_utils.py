# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion v3 model loading and preprocessing.

SD3 and SD3.5 share the StableDiffusion3Pipeline class from diffusers, but
they are released under different repositories and have different bringup
characteristics, so we keep their loaders isolated.
"""

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(
    pretrained_model_name: str, dtype: torch.dtype = torch.float32
) -> StableDiffusion3Pipeline:
    """Load a Stable Diffusion v3 pipeline.

    Args:
        pretrained_model_name: The HuggingFace repo name (under ``stabilityai/``).
        dtype: ``torch.dtype`` to load the pipeline weights in. Defaults to
            ``torch.float32``. Pass ``torch.bfloat16`` to load the weights
            directly in bf16 — the full fp32 pipeline (~30 GB, dominated by the
            T5-XXL text encoder) does not fit a 32 GB host, so loading at the
            target dtype avoids materializing the fp32 copy first.

    Returns:
        StableDiffusion3Pipeline: Loaded pipeline with all sub-modules set to
        eval mode and requires_grad disabled.
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        f"stabilityai/{pretrained_model_name}", torch_dtype=dtype
    )

    pipe.to("cpu")

    modules = [
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
        pipe.transformer,
        pipe.vae,
    ]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def calculate_shift(
    image_seq_len,
    base_image_seq_len,
    max_image_seq_len,
    base_shift,
    max_shift,
):
    """Calculate the dynamic shifting parameter ``mu`` for the SD3 scheduler."""
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return image_seq_len * m + b


def stable_diffusion_preprocessing_v3(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.0,
    num_inference_steps=1,
    num_images_per_prompt=1,
    clip_skip=None,
    max_sequence_length=256,
    do_classifier_free_guidance=True,
    mu=None,
):
    """Run the SD3 pipeline preprocessing (encode_prompt, latents, timestep).

    This mirrors :func:`stable_diffusion_preprocessing_v35` but is kept in this
    module so SD3 has no runtime dependency on the SD3.5 loader package.

    Returns:
        tuple: ``(latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds)``
        — the four tensors expected by ``SD3Transformer2DModel.forward``.
    """
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor

    pipe.check_inputs(
        prompt,
        None,  # prompt_2
        None,  # prompt_3
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=max_sequence_length,
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        device=device,
        clip_skip=clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

    num_channels_latents = pipe.transformer.config.in_channels
    shape = (
        num_images_per_prompt,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)

    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        # Use the latent spatial dims (not the original image height/width) to
        # match the reference SD3 pipeline implementation.
        _, _, latent_height, latent_width = latents.shape
        image_seq_len = (latent_height // pipe.transformer.config.patch_size) * (
            latent_width // pipe.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu

    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=None,
        **scheduler_kwargs,
    )

    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds


# ============================================================================
# SPMD shard specifications (MMDiT denoiser — SD3Transformer2DModel)
# ============================================================================

# (batch, model) mesh shapes by device count. Only the "model" axis is used as a
# real shard axis; "batch" stays data-parallel / replicated.
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8), 32: (8, 4)}
MESH_NAMES = (None, "model")


def shard_transformer_specs(transformer) -> dict:
    """Build tensor -> partition_spec dict for ``SD3Transformer2DModel``.

    Megatron-style tensor parallelism across the ``"model"`` mesh axis. SD3's
    MMDiT uses joint attention (an image stream + a text/context stream) per
    ``JointTransformerBlock``, mirroring Mochi's DiT, so the same column→row
    pairing applies. Heads (24) divide the model axis (2/4/8).

    Column-parallel (Q/K/V, FF up):  weight ("model", None), bias ("model",)
    Row-parallel    (out, FF down):  weight (None, "model"), bias replicated (None,)
    patch_embed / proj_out / norms / time & text embedders: replicated.
    """
    specs: dict = {}

    def col(linear):
        # Column-parallel: shard out-features (dim 0); bias is sharded too.
        specs[linear.weight] = ("model", None)
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = ("model",)

    def row(linear):
        # Row-parallel: shard in-features (dim 1); bias replicated (added after
        # the implicit all-reduce that follows the row-parallel matmul).
        specs[linear.weight] = (None, "model")
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = (None,)

    for block in transformer.transformer_blocks:
        attn = block.attn

        # Image-stream Q/K/V (column-parallel).
        col(attn.to_q)
        col(attn.to_k)
        col(attn.to_v)

        # Text/context-stream Q/K/V (column-parallel).
        if getattr(attn, "add_q_proj", None) is not None:
            col(attn.add_q_proj)
            col(attn.add_k_proj)
            col(attn.add_v_proj)

        # Output projections (row-parallel).
        row(attn.to_out[0])
        if getattr(attn, "to_add_out", None) is not None:
            row(attn.to_add_out)

        # FeedForward (GELU): net[0].proj is Linear(dim, inner) -> column-parallel,
        # net[2] is Linear(inner, dim) -> row-parallel.
        col(block.ff.net[0].proj)
        row(block.ff.net[2])
        if getattr(block, "ff_context", None) is not None:
            col(block.ff_context.net[0].proj)
            row(block.ff_context.net[2])

    return specs
