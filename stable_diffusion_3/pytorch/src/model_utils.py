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


def load_pipe(pretrained_model_name: str) -> StableDiffusion3Pipeline:
    """Load a Stable Diffusion v3 pipeline.

    Args:
        pretrained_model_name: The HuggingFace repo name (under ``stabilityai/``).

    Returns:
        StableDiffusion3Pipeline: Loaded pipeline with all sub-modules set to
        eval mode and requires_grad disabled.
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        f"stabilityai/{pretrained_model_name}", torch_dtype=torch.float32
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


class SD3TransformerWrapper(torch.nn.Module):
    """Adapt ``SD3Transformer2DModel`` to a plain-tensor, fixed-arg forward.

    ``SD3Transformer2DModel.forward`` expects positional args in the order
    ``(hidden_states, encoder_hidden_states, pooled_projections, timestep)``,
    whereas :func:`stable_diffusion_preprocessing_v3` emits
    ``(latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds)``.
    This wrapper accepts the preprocessing order and dispatches to the correct
    keyword arguments, so the raw transformer is never fed a mis-ordered
    ``timestep`` (which otherwise trips ``get_timestep_embedding``'s
    "Timesteps should be a 1d-array" assertion). It also pins
    ``return_dict=False`` so graph capture sees a pure tensor.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self, hidden_states, timestep, encoder_hidden_states, pooled_projections
    ):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            return_dict=False,
        )[0]


# ---------------------------------------------------------------------------
# T5-XXL text encoder component (text_encoder_3)
# ---------------------------------------------------------------------------
#
# SD3 Medium's third text encoder is a T5-v1.1-XXL ``T5EncoderModel`` (~4.7B
# params, d_model=4096, 24 layers, gated-gelu FFN). It is by far the heaviest
# component that the reference pipeline runs on CPU. This section exposes it as
# an independently loadable TT component, mirroring the text-encoder component
# loaders used by glm_image / lumina_image.

# Full HF repo id (``load_pipe`` prepends ``stabilityai/`` to the variant name;
# the component loaders below load the T5 subfolder directly).
REPO_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

# Default max_sequence_length used by StableDiffusion3Pipeline._get_t5_prompt_embeds.
T5_MAX_SEQ_LEN = 256
# T5 v1.1 SentencePiece vocab size (text_encoder_3/config.json: vocab_size).
T5_VOCAB_SIZE = 32128

# (batch, model) mesh shapes by device count — matches the sibling image loaders.
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def load_t5_text_encoder(dtype: torch.dtype):
    """Load the T5-XXL encoder (text_encoder_3) from the SD3 Medium repo.

    Loads only the ``text_encoder_3`` subfolder so we avoid materializing the
    transformer / VAE / other text encoders just to exercise the T5 tower.
    """
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder_3",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_t5_text_encoder_inputs(dtype: torch.dtype):
    """Inputs for the T5 encoder: ``[input_ids]``.

    Shapes match what ``_get_t5_prompt_embeds`` feeds the encoder: input_ids
    (1, 256) int64. The reference pipeline calls
    ``self.text_encoder_3(text_input_ids)[0]`` with no attention_mask, so the
    wrapper below takes only ``input_ids`` and we return a single tensor.
    """
    input_ids = torch.randint(0, T5_VOCAB_SIZE, (1, T5_MAX_SEQ_LEN), dtype=torch.long)
    return [input_ids]


class T5TextEncoderWrapper(torch.nn.Module):
    """Run ``T5EncoderModel`` as a stateless encoder returning a plain tensor.

    Pins ``return_dict=False`` so graph capture sees a pure tensor
    (last_hidden_state) rather than a ``BaseModelOutput`` dataclass. This
    mirrors how ``StableDiffusion3Pipeline._get_t5_prompt_embeds`` consumes the
    encoder: ``self.text_encoder_3(text_input_ids)[0]``.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        return self.encoder(input_ids=input_ids, return_dict=False)[0]


def shard_t5_text_encoder_specs(encoder) -> dict:
    """Megatron-style tensor-parallel shard specs for ``T5EncoderModel``.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, wi_0, wi_1): ("model", "batch")
    Row-parallel   (o, wo):                ("batch", "model")

    ``encoder`` is the raw ``T5EncoderModel``. Layer norms and the relative
    attention bias are left replicated. On a single device (mesh (1, 1)) these
    specs are a no-op; they only take effect for the tensor-parallel runs.
    """
    specs = {encoder.shared.weight: (None, "batch")}

    # encoder.encoder is the T5Stack holding the transformer blocks.
    for block in encoder.encoder.block:
        attn = block.layer[0].SelfAttention
        specs[attn.q.weight] = ("model", "batch")
        specs[attn.k.weight] = ("model", "batch")
        specs[attn.v.weight] = ("model", "batch")
        specs[attn.o.weight] = ("batch", "model")

        ff = block.layer[1].DenseReluDense
        specs[ff.wi_0.weight] = ("model", "batch")
        specs[ff.wi_1.weight] = ("model", "batch")
        specs[ff.wo.weight] = ("batch", "model")

    return specs


# ---------------------------------------------------------------------------
# VAE decoder component (vae)
# ---------------------------------------------------------------------------
#
# SD3 Medium's VAE is a 16-latent-channel ``AutoencoderKL``. The reference
# pipeline runs the decoder on CPU to turn denoised latents into pixels. This
# section exposes the decoder half as an independently compilable TT component,
# mirroring the VAE component loaders used by glm_image / lumina_image.

# AutoencoderKL config: latent_channels=16, sample_size=1024, downsample x8.
VAE_LATENT_CHANNELS = 16
VAE_SAMPLE_SIZE = 1024
VAE_SCALE_FACTOR = 8
# Latent spatial size fed to the decoder for a 1024x1024 image (1024 // 8).
VAE_LATENT_HW = VAE_SAMPLE_SIZE // VAE_SCALE_FACTOR  # 128


def load_sd3_vae(dtype: torch.dtype):
    """Load the SD3 Medium ``AutoencoderKL`` from the ``vae`` subfolder."""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_sd3_vae_inputs(dtype: torch.dtype):
    """Inputs for the VAE decoder: ``[z]``.

    ``z`` is a denoised latent of shape (1, 16, 128, 128) for a 1024x1024
    image — the tensor the reference pipeline feeds into ``vae.decode``.
    """
    z = torch.randn(1, VAE_LATENT_CHANNELS, VAE_LATENT_HW, VAE_LATENT_HW, dtype=dtype)
    return [z]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of ``AutoencoderKL`` as ``(z) -> tensor``.

    The default ``vae(z)`` runs encode+decode and returns a ``ModelOutput``.
    This wrapper calls ``decode`` directly and unwraps to a plain tensor so
    graph capture sees the reconstructed image tensor.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]
