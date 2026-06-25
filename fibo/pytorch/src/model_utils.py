# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders, wrappers and input builders for BRIA FIBO.

Model: briaai/FIBO  (custom ``BriaFiboPipeline``, diffusers >= 0.38) — a gated,
8B-parameter DiT flow-matching text-to-image model trained on structured-JSON
captions (paper: arXiv 2511.06876).

FIBO is a multi-component pipeline, so — like ``omnigen`` — each independently
compilable component is exposed as its own ``ModelVariant`` rather than wrapping
the whole pipeline in one graph:

  - text_encoder: ``SmolLM3ForCausalLM`` — encodes the JSON prompt. The pipeline
                  concatenates its last two hidden states into the 4096-dim
                  ``encoder_hidden_states`` and also forwards *every* hidden
                  state (one per transformer block) as ``text_encoder_layers``.
  - transformer:  ``BriaFiboTransformer2DModel`` — the Flux-style MMDiT denoiser
                  (8 dual + 38 single layers, 24 heads x 128 head_dim). The heavy
                  per-step compute and the bringup's primary target.
  - vae:          ``AutoencoderKLWan`` — Wan 3D causal-conv VAE; only the decoder
                  (latent -> image) is exposed.

The ``FlowMatchEulerDiscreteScheduler`` and the tokenizer carry no learnable
weights and stay in host Python (composite step), so they are not variants.

All shapes below are for the model's native/default 1024x1024 generation with
classifier-free guidance (``guidance_scale=5`` -> CFG batch = 2), derived by
tracing the live ``BriaFiboPipeline`` at ``num_inference_steps=1``:

  vae_scale_factor = 16, do_patching = False
    latent grid  = 1024 // 16 = 64  ->  64 * 64 = 4096 image tokens
    packed latent channels = transformer.in_channels = 48

The gated repo requires accepting the bria-fibo license on HF and an
authenticated ``HF_TOKEN`` before the weights can be fetched.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "briaai/FIBO"
DTYPE = torch.bfloat16

# A stub structured-JSON prompt (FIBO is trained on structured captions). Used
# by the composite generation step; the tokenizer/SmolLM3 path is identical
# whether or not the JSON is semantically meaningful.
BRINGUP_PROMPT = (
    '{"short_description":"a hyper-detailed, ultra-fluffy owl perched in '
    'moonlit trees, looking at the camera with wide expressive eyes",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)

# ---------------------------------------------------------------------------
# Native-resolution shape constants (1024 x 1024, CFG batch = 2)
# ---------------------------------------------------------------------------

IMAGE_H = 1024
IMAGE_W = 1024
VAE_SCALE_FACTOR = 16
LATENT_H = IMAGE_H // VAE_SCALE_FACTOR  # 64
LATENT_W = IMAGE_W // VAE_SCALE_FACTOR  # 64
NUM_IMAGE_TOKENS = LATENT_H * LATENT_W  # 4096

PACKED_LATENT_CHANNELS = 48  # transformer.config.in_channels (do_patching=False)
VAE_Z_DIM = 48               # AutoencoderKLWan latent channels
TEXT_HIDDEN = 2048           # SmolLM3 hidden_size
ENCODER_HIDDEN = 4096        # transformer joint_attention_dim (cat of 2 layers)
NUM_TRANSFORMER_LAYERS = 46  # 8 dual + 38 single -> len(text_encoder_layers)

# Representative text-token length for a structured-JSON prompt (the example
# "generate.json" prompt tokenises to ~509 tokens after special tokens).
TEXT_SEQ_LEN = 509
TOTAL_SEQ_LEN = TEXT_SEQ_LEN + NUM_IMAGE_TOKENS  # 4605 (text + image attn span)

CFG_BATCH = 2  # classifier-free guidance doubles the batch
VOCAB_SIZE = 128256


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Load SmolLM3ForCausalLM from the text_encoder subfolder."""
    from transformers import SmolLM3ForCausalLM

    return SmolLM3ForCausalLM.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).eval()


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load BriaFiboTransformer2DModel from the transformer subfolder."""
    from diffusers import BriaFiboTransformer2DModel

    return BriaFiboTransformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """Load AutoencoderKLWan from the vae subfolder."""
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
    ).eval()


def load_pipe(pretrained_model_name: str = REPO_ID, dtype_override=None):
    """Load the full FIBO pipeline (used by the host-Python composite step)."""
    from diffusers import BriaFiboPipeline

    kwargs = {}
    if dtype_override is not None:
        kwargs["torch_dtype"] = dtype_override
    pipe = BriaFiboPipeline.from_pretrained(pretrained_model_name, **kwargs)
    return pipe


# ---------------------------------------------------------------------------
# Wrapper modules — flatten each component to positional tensor I/O so the
# auto-runner (DynamicTorchModelTester) can compile them without glue.
# ---------------------------------------------------------------------------


class TextEncoderWrapper(torch.nn.Module):
    """SmolLM3 text encoder -> the pipeline's 4096-dim ``encoder_hidden_states``.

    Reproduces ``BriaFiboPipeline.get_prompt_embeds``: run with
    ``output_hidden_states=True`` and concatenate the last two hidden states.
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hs = out.hidden_states
        return torch.cat([hs[-1], hs[-2]], dim=-1)


class TransformerWrapper(torch.nn.Module):
    """BriaFiboTransformer2DModel with positional-tensor-only I/O.

    The pipeline passes ``text_encoder_layers`` (a list of one hidden-state
    tensor per transformer block) and ``joint_attention_kwargs={'attention_mask'}``.
    Here the per-layer tensors arrive pre-stacked as ``layers_stacked``
    ``[num_layers, B, seq, hidden]`` and are unbound back into a list, and the
    attention mask is a plain tensor argument.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        layers_stacked,
        txt_ids,
        img_ids,
        attention_mask,
    ):
        text_encoder_layers = list(torch.unbind(layers_stacked, dim=0))
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            text_encoder_layers=text_encoder_layers,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs={"attention_mask": attention_mask},
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only the AutoencoderKLWan decoder as (z) -> image tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Component input builders (native 1024x1024 shapes)
# ---------------------------------------------------------------------------


def load_text_encoder_inputs(dtype: torch.dtype = DTYPE):
    """[input_ids (1, TEXT_SEQ_LEN) long, attention_mask (1, TEXT_SEQ_LEN) long].

    dtype only affects float tensors; ids/mask stay integer.
    """
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(
        0, VOCAB_SIZE, (1, TEXT_SEQ_LEN), generator=g, dtype=torch.long
    )
    attention_mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)
    return [input_ids, attention_mask]


def load_transformer_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for TransformerWrapper at native 1024x1024 / CFG batch 2.

    Returns [hidden_states, timestep, encoder_hidden_states, layers_stacked,
             txt_ids, img_ids, attention_mask].
    """
    g = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(
        CFG_BATCH, NUM_IMAGE_TOKENS, PACKED_LATENT_CHANNELS, generator=g, dtype=dtype
    )
    timestep = torch.zeros(CFG_BATCH, dtype=dtype)
    encoder_hidden_states = torch.randn(
        CFG_BATCH, TEXT_SEQ_LEN, ENCODER_HIDDEN, generator=g, dtype=dtype
    )
    layers_stacked = torch.randn(
        NUM_TRANSFORMER_LAYERS, CFG_BATCH, TEXT_SEQ_LEN, TEXT_HIDDEN,
        generator=g, dtype=dtype,
    )
    # Flux-style positional ids: txt ids are zeros; img ids index the latent grid.
    txt_ids = torch.zeros(TEXT_SEQ_LEN, 3, dtype=dtype)
    img_ids = torch.zeros(LATENT_H, LATENT_W, 3, dtype=dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(LATENT_H)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(LATENT_W)[None, :]
    img_ids = img_ids.reshape(LATENT_H * LATENT_W, 3)
    # Joint (text + image) self-attention mask, head-broadcast: [B, 1, S, S].
    attention_mask = torch.ones(
        CFG_BATCH, 1, TOTAL_SEQ_LEN, TOTAL_SEQ_LEN, dtype=dtype
    )
    return [
        hidden_states,
        timestep,
        encoder_hidden_states,
        layers_stacked,
        txt_ids,
        img_ids,
        attention_mask,
    ]


def load_vae_decoder_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic Wan-VAE latent for VAEDecoderWrapper: [z (1, 48, 1, 64, 64)].

    Shape = [B, z_dim, T=1 (single image frame), LATENT_H, LATENT_W].
    """
    g = torch.Generator().manual_seed(0)
    z = torch.randn(1, VAE_Z_DIM, 1, LATENT_H, LATENT_W, generator=g, dtype=dtype)
    return [z]
