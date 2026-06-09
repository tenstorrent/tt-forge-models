# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Tongyi-MAI/Z-Image (ZImagePipeline).
"""

import torch

REPO_ID = "Tongyi-MAI/Z-Image"
DTYPE = torch.bfloat16

# ZImagePipeline inference defaults
PROMPT = "A red cube on a white table, studio lighting, sharp focus."
NEGATIVE_PROMPT = "blurry, text, watermark"
HEIGHT = 1280
WIDTH = 720
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 4.0
SEED = 42
CFG_NORMALIZATION = False
MAX_SEQUENCE_LENGTH = 512

VAE_SCALE_FACTOR = 8
VAE_SPATIAL_ALIGN = VAE_SCALE_FACTOR * 2
LATENT_CHANNELS = 16

TEXT_TOKEN_MAX_LEN = MAX_SEQUENCE_LENGTH
TEXT_EMBED_DIM = 2560
TEXT_ENCODER_NUM_LAYERS = 36
QWEN_VOCAB_SIZE = 151936

TRANSFORMER_NUM_LAYERS = 30
TRANSFORMER_NUM_REFINER_LAYERS = 2
TRANSFORMER_IN_CHANNELS = 16
TRANSFORMER_CAP_FEAT_DIM = 2560

# (batch, model) mesh shapes by device count.
# Megatron TP shards only on "model"; keep batch dim at 1 for counts < 32 so
# Shardy does not reshard activations onto the batch axis (collective_permute;
# tt-xla#2496 / tt-mlir#3370). Galaxy (32) uses (8, 4).
MESH_SHAPES = {32: (8, 4), 8: (1, 8), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def latent_hw_from_pixels(height: int = HEIGHT, width: int = WIDTH) -> tuple[int, int]:
    latent_h = 2 * (int(height) // VAE_SPATIAL_ALIGN)
    latent_w = 2 * (int(width) // VAE_SPATIAL_ALIGN)
    return latent_h, latent_w


LATENT_H, LATENT_W = latent_hw_from_pixels()


def make_latent_inputs(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1, LATENT_CHANNELS, LATENT_H, LATENT_W, dtype=dtype, generator=gen
    )


def load_vae(dtype: torch.dtype = DTYPE):
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")


def tokenize_prompt(
    prompt: str = PROMPT,
    *,
    max_sequence_length: int = TEXT_TOKEN_MAX_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = load_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids, text_inputs.attention_mask


def encode_prompt_hidden_states(
    encoder: torch.nn.Module,
    prompt: str = PROMPT,
    *,
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    input_ids, attention_mask = tokenize_prompt(prompt)
    with torch.no_grad():
        hidden = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            output_hidden_states=True,
        ).hidden_states[-2]
    valid = hidden[0, attention_mask[0].bool()]
    return valid.unsqueeze(0).to(dtype)


def load_text_encoder(dtype: torch.dtype = DTYPE):
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()
    if len(encoder.layers) != TEXT_ENCODER_NUM_LAYERS:
        raise ValueError(
            f"Expected {TEXT_ENCODER_NUM_LAYERS} layers, got {len(encoder.layers)}"
        )
    return encoder


class Qwen3TextEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-2]


def load_transformer(dtype: torch.dtype = DTYPE):
    from diffusers import ZImageTransformer2DModel

    transformer = ZImageTransformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()
    if len(transformer.layers) != TRANSFORMER_NUM_LAYERS:
        raise ValueError(
            f"Expected {TRANSFORMER_NUM_LAYERS} layers, got {len(transformer.layers)}"
        )
    if len(transformer.noise_refiner) != TRANSFORMER_NUM_REFINER_LAYERS:
        raise ValueError(
            f"Expected {TRANSFORMER_NUM_REFINER_LAYERS} noise refiner layers, "
            f"got {len(transformer.noise_refiner)}"
        )
    return transformer


class ZImageTransformerWrapper(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cap_feats: torch.Tensor,
    ) -> torch.Tensor:
        x_list = list(latents.unsqueeze(2).unbind(dim=0))
        cap_list = [cap_feats[i] for i in range(cap_feats.shape[0])]
        t = timestep.reshape(-1).to(dtype=latents.dtype, device=latents.device)
        out_list = self.transformer(x_list, t, cap_list, return_dict=False)[0]
        return torch.stack([o.float() for o in out_list], dim=0).squeeze(2)


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(dtype=self.vae.dtype)
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications (transformer only — ~6B params, needs 2+ chips)
# ---------------------------------------------------------------------------


def _set_shard_spec(specs: dict, tensor: torch.Tensor, spec: tuple) -> None:
    """Record a partition spec whose length matches ``tensor.ndim`` (None = replicate)."""
    if len(spec) != tensor.ndim:
        raise ValueError(
            f"Partition spec length {len(spec)} != tensor rank {tensor.ndim} "
            f"(shape {tuple(tensor.shape)}): {spec}"
        )
    specs[tensor] = spec


def _replicate_spec(tensor: torch.Tensor) -> tuple:
    """Fully replicated spec for any rank (pad tokens, small params)."""
    return (None,) * tensor.ndim


def _shard_zimage_block(block, specs: dict, *, has_modulation: bool) -> None:
    """Megatron-style sharding for a single ZImageTransformerBlock."""
    attn = block.attention
    for proj_name in ("to_q", "to_k", "to_v"):
        proj = getattr(attn, proj_name)
        specs[proj.weight] = ("model", None)
        if proj.bias is not None:
            specs[proj.bias] = ("model",)

    to_out = (
        attn.to_out[0]
        if isinstance(attn.to_out, (torch.nn.Sequential, torch.nn.ModuleList))
        else attn.to_out
    )
    specs[to_out.weight] = (None, "model")
    if to_out.bias is not None:
        specs[to_out.bias] = (None,)

    if getattr(attn, "norm_q", None) is not None:
        specs[attn.norm_q.weight] = ("model",)
    if getattr(attn, "norm_k", None) is not None:
        specs[attn.norm_k.weight] = ("model",)

    ff = block.feed_forward
    specs[ff.w1.weight] = ("model", None)
    specs[ff.w3.weight] = ("model", None)
    specs[ff.w2.weight] = (None, "model")

    # Block RMSNorms are left unmarked (replicated) — do not force (None,) specs
    # that can trigger cross-axis reshards when hidden states are model-sharded.

    if has_modulation and hasattr(block, "adaLN_modulation"):
        lin = block.adaLN_modulation[0]
        specs[lin.weight] = ("model", None)
        if lin.bias is not None:
            specs[lin.bias] = ("model",)


def _shard_final_layer(layer, specs: dict) -> None:
    specs[layer.linear.weight] = (None, "model")
    if layer.linear.bias is not None:
        specs[layer.linear.bias] = (None,)
    ada_lin = layer.adaLN_modulation[1]
    specs[ada_lin.weight] = ("model", None)
    if ada_lin.bias is not None:
        specs[ada_lin.bias] = ("model",)


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for ZImageTransformer2DModel.

    Mesh axes: ("batch", "model")
    Column-parallel (Q, K, V, FFN w1/w3): ("model", None)
    Row-parallel   (O, FFN w2):           (None, "model")
    """
    specs = {}

    for embedder in transformer.all_x_embedder.values():
        specs[embedder.weight] = ("model", None)
        if embedder.bias is not None:
            specs[embedder.bias] = ("model",)

    for final_layer in transformer.all_final_layer.values():
        _shard_final_layer(final_layer, specs)

    mlp = transformer.t_embedder.mlp
    specs[mlp[0].weight] = ("model", None)
    if mlp[0].bias is not None:
        specs[mlp[0].bias] = ("model",)
    specs[mlp[2].weight] = (None, "model")
    if mlp[2].bias is not None:
        specs[mlp[2].bias] = (None,)

    cap_lin = transformer.cap_embedder[1]
    specs[cap_lin.weight] = ("model", None)
    if cap_lin.bias is not None:
        specs[cap_lin.bias] = ("model",)

    for block in transformer.noise_refiner:
        _shard_zimage_block(block, specs, has_modulation=True)
    for block in transformer.context_refiner:
        _shard_zimage_block(block, specs, has_modulation=False)
    for block in transformer.layers:
        _shard_zimage_block(block, specs, has_modulation=True)

    # Pad tokens are (1, dim); must not use 1-axis specs like ("model",).
    for pad_name in ("x_pad_token", "cap_pad_token", "siglip_pad_token"):
        pad = getattr(transformer, pad_name, None)
        if pad is not None:
            _set_shard_spec(specs, pad, _replicate_spec(pad))

    return specs


def shard_transformer_input_specs(args) -> dict:
    """Replicate forward-pass inputs so Shardy does not reshard them onto batch."""
    specs = {}
    for tensor in args:
        if isinstance(tensor, torch.Tensor):
            _set_shard_spec(specs, tensor, _replicate_spec(tensor))
    return specs
