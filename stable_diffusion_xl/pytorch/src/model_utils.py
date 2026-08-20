# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion XL model loading and processing.
"""

from typing import List, Optional, Tuple, Union
import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(variant):
    """Load Stable Diffusion XL pipeline.

    Args:
        variant: Model variant name

    Returns:
        DiffusionPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = DiffusionPipeline.from_pretrained(
        f"stabilityai/{variant}", torch_dtype=torch.float32
    )
    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]

    # Move the pipeline to CPU
    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def stable_diffusion_preprocessing_xl(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=5.0,
    num_inference_steps=50,
    timesteps=None,
    sigmas=None,
    eta=0.0,
    num_images_per_prompt=1,
    height=None,
    width=None,
    clip_skip=None,
    original_size=None,
    target_size=None,
    cross_attention_kwargs=None,
    guidance_rescale=0.0,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_target_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    **kwargs,
):
    """Preprocess inputs for Stable Diffusion XL model.

    Args:
        pipe: Stable Diffusion XL pipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 5.0)
        num_inference_steps: Number of inference steps (default: 50)
        timesteps: Custom timesteps (optional)
        sigmas: Custom sigmas (optional)
        eta: Eta parameter (default: 0.0)
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (optional, uses default if None)
        width: Image width (optional, uses default if None)
        clip_skip: CLIP skip layers (optional)
        original_size: Original size tuple (optional)
        target_size: Target size tuple (optional)
        cross_attention_kwargs: Cross attention kwargs (optional)
        guidance_rescale: Guidance rescale factor (default: 0.0)
        crops_coords_top_left: Crop coordinates (default: (0, 0))
        negative_original_size: Negative original size (optional)
        negative_target_size: Negative target size (optional)
        negative_crops_coords_top_left: Negative crop coordinates (default: (0, 0))
        **kwargs: Additional keyword arguments

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds, timestep_cond, added_cond_kwargs, add_time_ids)
    """
    # Set default height and width
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # Check inputs
    pipe.check_inputs(
        prompt,
        None,  # prompt_2 (if applicable)
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_steps=None,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    # 1. Encode the prompt
    do_classifier_free_guidance = True
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,  # Assume classifier-free guidance
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
    )
    # 2. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        timesteps=timesteps,
        sigmas=sigmas,
    )

    # 3. Prepare latent variables
    if isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    num_channels_latents = pipe.unet.config.in_channels
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    torch.manual_seed(42)
    latents = torch.randn(
        (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipe._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
    ip_adapter_image = None
    ip_adapter_image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            do_classifier_free_guidance,
        )
    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(
            batch_size * num_images_per_prompt
        )
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        added_cond_kwargs["image_embeds"] = image_embeds

    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, timesteps[0]
    )

    return (
        latent_model_input,
        timesteps,
        prompt_embeds,
        timestep_cond,
        added_cond_kwargs,
        add_time_ids,
    )


# --------------------------------------------------------------------------- #
# CLIP text encoders as independently compilable TT components.
#
# SDXL conditions its UNet on TWO CLIP towers (diffusers
# ``StableDiffusionXLPipeline.encode_prompt``): it walks
# ``[text_encoder, text_encoder_2]`` and, for each tower, takes
# ``hidden_states[-2]`` — the penultimate layer, which SDXL always indexes —
# concatenating the two along the feature axis into ``prompt_embeds``. The
# pooled embedding is taken from ``prompt_embeds[0]`` only when it is 2-D,
# which is true for the projection output of ``text_encoder_2`` alone; that
# tensor becomes ``add_text_embeds`` in ``added_cond_kwargs``. CLIP-L's ``[0]``
# is its 3-D ``last_hidden_state`` and is never consumed.
#
# Hence the asymmetry between the two wrappers below: CLIP-L returns one
# tensor, CLIP-G returns two.
# --------------------------------------------------------------------------- #

# CLIP context length (both tokenizers' model_max_length, and both configs'
# max_position_embeddings) and vocab size — identical for the two towers.
CLIP_MAX_SEQ_LEN = 77
CLIP_VOCAB_SIZE = 49408

# Subfolder in the SDXL base repo per CLIP tower.
CLIP_L_SUBFOLDER = "text_encoder"
CLIP_G_SUBFOLDER = "text_encoder_2"

# Tokenizer subfolder paired with each encoder subfolder.
CLIP_TOKENIZER_SUBFOLDERS = {
    CLIP_L_SUBFOLDER: "tokenizer",
    CLIP_G_SUBFOLDER: "tokenizer_2",
}


def _repo_id(pretrained_model_name: str) -> str:
    """``"stable-diffusion-xl-base-1.0"`` -> the HuggingFace repo id."""
    return f"stabilityai/{pretrained_model_name}"


def load_clip_text_encoder(
    pretrained_model_name: str, dtype: torch.dtype, subfolder: str
):
    """Load one CLIP tower from the SDXL base repo.

    Args:
        pretrained_model_name: variant name, e.g. ``"stable-diffusion-xl-base-1.0"``.
        dtype: torch dtype for the returned module.
        subfolder: ``"text_encoder"`` (CLIP-L) or ``"text_encoder_2"`` (CLIP-G).

    The class matches each subfolder's ``config.json`` ``architectures``:
    ``CLIPTextModel`` for CLIP-L, ``CLIPTextModelWithProjection`` for CLIP-G.
    CLIP-G must keep its projection head — that head produces SDXL's
    ``add_text_embeds`` — while CLIP-L's projection is never used, so loading it
    as the bare model avoids materializing an unused head.

    Only the requested subfolder is loaded, so exercising one tower never
    materializes the UNet, the VAE or the other tower.
    """
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    cls = (
        CLIPTextModelWithProjection if subfolder == CLIP_G_SUBFOLDER else CLIPTextModel
    )
    return cls.from_pretrained(
        _repo_id(pretrained_model_name),
        subfolder=subfolder,
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_clip_text_encoder_inputs(
    pretrained_model_name: str, prompt: str, subfolder: str
):
    """Inputs for a CLIP tower: ``[input_ids]`` of shape ``(1, 77)`` int64.

    Tokenized by the tower's own tokenizer with
    ``padding="max_length", truncation=True``, i.e. exactly what
    ``encode_prompt`` feeds the encoder. Token ids are always int64, so there is
    no dtype argument.
    """
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        _repo_id(pretrained_model_name),
        subfolder=CLIP_TOKENIZER_SUBFOLDERS[subfolder],
    )
    input_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=CLIP_MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return [input_ids.to(torch.long)]


class CLIPTextEncoderWrapper(torch.nn.Module):
    """Run SDXL's CLIP-L tower as ``(input_ids) -> penultimate_hidden``.

    Returns the single tensor the pipeline consumes from this tower: the
    penultimate hidden state, which is concatenated with CLIP-G's along the
    feature axis into ``prompt_embeds``. Returning a plain tensor keeps the
    ``BaseModelOutputWithPooling`` dataclass out of graph capture.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        out = self.encoder(
            input_ids=input_ids, output_hidden_states=True, return_dict=True
        )
        return out.hidden_states[-2]


class CLIPGTextEncoderWrapper(torch.nn.Module):
    """Run SDXL's CLIP-G tower as ``(input_ids) -> (penultimate_hidden, pooled)``.

    Returns both tensors the pipeline consumes from this tower, so a compiled
    run covers both paths through it: the penultimate hidden state (the second
    half of ``prompt_embeds``) and the projected pooled embedding (SDXL's
    ``add_text_embeds``). ``pooled`` comes from the projection head, which is
    why this tower is loaded as ``CLIPTextModelWithProjection``.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        out = self.encoder(
            input_ids=input_ids, output_hidden_states=True, return_dict=True
        )
        return out.hidden_states[-2], out.text_embeds
