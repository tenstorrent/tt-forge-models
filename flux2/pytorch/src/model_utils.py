# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for black-forest-labs/FLUX.2-dev (Flux2Pipeline).

Bringup resolution: 128x128 pixels (matches TT component-test plan).
"""

import torch

REPO_ID = "black-forest-labs/FLUX.2-dev"
DTYPE = torch.bfloat16

# Flux2Pipeline defaults aligned with diffusers/pipelines/flux2/pipeline_flux2.py
PROMPT = (
    "Realistic macro photograph of a hermit crab using a soda can as its shell, "
    "partially emerging from the can, on a sunlit beach."
)
HEIGHT = 128
WIDTH = 128
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 4.0
SEED = 42
MAX_SEQUENCE_LENGTH = 512
TEXT_ENCODER_OUT_LAYERS = (10, 20, 30)

VAE_SCALE_FACTOR = 8
VAE_SPATIAL_ALIGN = VAE_SCALE_FACTOR * 2  # 16

# Transformer config (FLUX.2-dev)
TRANSFORMER_IN_CHANNELS = 128
JOINT_ATTENTION_DIM = 15360
NUM_LATENT_CHANNELS = TRANSFORMER_IN_CHANNELS // 4  # 32

# Latent geometry at 128x128 output (see Flux2Pipeline.prepare_latents)
def _latent_grid_hw(height: int = HEIGHT, width: int = WIDTH) -> tuple[int, int]:
    h = 2 * (int(height) // VAE_SPATIAL_ALIGN)
    w = 2 * (int(width) // VAE_SPATIAL_ALIGN)
    return h // 2, w // 2


LATENT_GRID_H, LATENT_GRID_W = _latent_grid_hw()
LATENT_PACKED_SEQ = LATENT_GRID_H * LATENT_GRID_W
# Packed latent channels before VAE decode (num_latents_channels * 4)
VAE_LATENT_C = TRANSFORMER_IN_CHANNELS

# (batch, model) mesh shapes by device count.
#
# The shard specs below partition weights only along the "model" axis, and the
# component tests run a single sample (batch == 1). A non-trivial "batch" axis
# therefore shards nothing and merely replicates the full model across that axis,
# which caps effective model-parallelism and OOMs large encoders/transformers
# (e.g. (2, 4) gives only 4-way weight sharding on 8 chips → the ~47 GB bf16
# text encoder needs ~14 GB/device and runs out of DRAM). Put every device on
# the "model" axis so weights shard across all of them.
MESH_SHAPES = {32: (1, 32), 8: (1, 8), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def load_text_encoder(dtype: torch.dtype = DTYPE):
    from transformers import Mistral3ForConditionalGeneration

    return Mistral3ForConditionalGeneration.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_tokenizer():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(REPO_ID, subfolder="tokenizer")


def tokenize_prompt(
    prompt: str = PROMPT,
    *,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    from diffusers.pipelines.flux2.pipeline_flux2 import SYSTEM_MESSAGE, format_input

    tokenizer = load_tokenizer()
    messages = format_input(prompts=[prompt], system_message=SYSTEM_MESSAGE)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    return inputs["input_ids"], inputs["attention_mask"]


def load_transformer(dtype: torch.dtype = DTYPE):
    from diffusers import Flux2Transformer2DModel

    return Flux2Transformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    from diffusers import AutoencoderKLFlux2

    return AutoencoderKLFlux2.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


class Mistral3TextEncoderWrapper(torch.nn.Module):
    """Match Flux2Pipeline._get_mistral_3_small_prompt_embeds (tensor-in / tensor-out)."""

    def __init__(
        self,
        text_encoder,
        hidden_states_layers: tuple[int, ...] = TEXT_ENCODER_OUT_LAYERS,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.hidden_states_layers = hidden_states_layers

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        out = torch.stack(
            [output.hidden_states[k] for k in self.hidden_states_layers], dim=1
        )
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        return out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )


class Flux2TransformerWrapper(torch.nn.Module):
    """Single denoise step: same signature as Flux2Transformer2DModel.forward."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)


class Flux2VAEDecoderWrapper(torch.nn.Module):
    """Decode path after unpack + batch-norm denorm (Flux2Pipeline.__call__)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            device=latents.device, dtype=latents.dtype
        )
        std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(device=latents.device, dtype=latents.dtype)
        latents = latents * std + mean
        latents = _unpatchify_latents(latents)
        return self.vae.decode(latents, return_dict=False)[0]


def make_packed_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1, LATENT_PACKED_SEQ, TRANSFORMER_IN_CHANNELS, dtype=dtype, generator=gen
    )


def make_prompt_embeds(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    encoder = load_text_encoder(dtype)
    input_ids, attention_mask = tokenize_prompt()
    wrapper = Mistral3TextEncoderWrapper(encoder)
    with torch.no_grad():
        return wrapper(input_ids, attention_mask)


def prepare_text_ids(batch_size: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.zeros(seq_len, 4)
    coords[:, 3] = torch.arange(seq_len, dtype=coords.dtype)
    return coords.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)


def prepare_latent_image_ids(
    batch_size: int, height: int, width: int, dtype: torch.dtype
) -> torch.Tensor:
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    coords = torch.cartesian_prod(t, h, w, l)
    return coords.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)


def make_vae_decoder_input(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Normalized latent grid (1, 128, 8, 8) — same layout as Flux2Pipeline pre-decode."""
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1,
        VAE_LATENT_C,
        LATENT_GRID_H,
        LATENT_GRID_W,
        dtype=dtype,
        generator=gen,
    )


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------


def _add_shard_spec(specs: dict, param: torch.Tensor | None, spec: tuple) -> None:
    """Register a partition spec only for real parameters (skip None weights/biases)."""
    if param is not None:
        specs[param] = spec


def _resolve_text_transformer(encoder):
    """Descend wrapper modules until reaching the decoder stack that owns `.layers`.

    Mistral3ForConditionalGeneration nests the actual transformer as
    ``encoder.model.language_model`` (Mistral3Model wraps a vision tower +
    language model). The decoder block stack — the module that exposes
    ``layers``/``embed_tokens`` — lives below those wrappers, so walk through the
    common wrapper attributes until we find it. Returns the wrapper found last if
    no `.layers` exists (e.g. a bare decoder passed directly).

    The descent has no fixed depth: it keeps unwrapping until it reaches a module
    with ``.layers`` or one that exposes none of the known wrapper attributes. A
    ``visited`` set guards against pathological self-referential modules.
    """
    module = encoder
    visited = {id(module)}
    while not hasattr(module, "layers"):
        for attr in ("language_model", "model", "text_model"):
            inner = getattr(module, attr, None)
            if inner is not None and id(inner) not in visited:
                module = inner
                visited.add(id(module))
                break
        else:
            break
    return module


def shard_text_encoder_specs(encoder) -> dict:
    """Shard Mistral3 / language-model blocks (column/row parallel)."""
    specs = {}
    language_model = _resolve_text_transformer(encoder)

    if hasattr(language_model, "embed_tokens"):
        specs[language_model.embed_tokens.weight] = (None, None)

    layers = getattr(language_model, "layers", None)
    if layers is None:
        return specs

    for layer in layers:
        sa = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(sa, proj_name)
            specs[proj.weight] = ("model", None)
            if proj.bias is not None:
                specs[proj.bias] = ("model",)
        specs[sa.o_proj.weight] = (None, "model")
        if sa.o_proj.bias is not None:
            specs[sa.o_proj.bias] = (None,)

        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", None)
        specs[mlp.up_proj.weight] = ("model", None)
        specs[mlp.down_proj.weight] = (None, "model")

        specs[layer.input_layernorm.weight] = (None,)
        specs[layer.post_attention_layernorm.weight] = (None,)

    if hasattr(language_model, "norm"):
        specs[language_model.norm.weight] = (None,)

    return specs


def _shard_flux2_joint_attention(attn, specs: dict) -> None:
    """Flux2Attention in dual-stream transformer_blocks."""
    for proj_name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        if hasattr(attn, proj_name) and getattr(attn, proj_name) is not None:
            proj = getattr(attn, proj_name)
            _add_shard_spec(specs, proj.weight, ("model", None))
            _add_shard_spec(specs, proj.bias, ("model",))

    to_out = attn.to_out
    if isinstance(to_out, torch.nn.ModuleList) and len(to_out) > 0:
        out_proj = to_out[0]
    elif isinstance(to_out, torch.nn.Linear):
        out_proj = to_out
    else:
        out_proj = None
    if out_proj is not None:
        _add_shard_spec(specs, out_proj.weight, (None, "model"))
        _add_shard_spec(specs, out_proj.bias, (None,))

    if hasattr(attn, "to_add_out") and attn.to_add_out is not None:
        _add_shard_spec(specs, attn.to_add_out.weight, (None, "model"))
        _add_shard_spec(specs, attn.to_add_out.bias, (None,))

    # QK-RMSNorm weights are shape (head_dim,) and apply per-head over head_dim.
    # Sharding splits *heads* across the "model" axis (each device keeps full
    # head_dim), so these must be REPLICATED — sharding them ("model",) hands each
    # device only head_dim/num_devices of the norm weights and corrupts the
    # normalization. (Harmless on random init where every weight is 1.0, but
    # catastrophic with trained weights → attention output decorrelates.)
    for norm_name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        norm = getattr(attn, norm_name, None)
        if norm is not None:
            _add_shard_spec(specs, getattr(norm, "weight", None), (None,))


def _shard_flux2_parallel_attention(attn, specs: dict) -> None:
    """Flux2ParallelSelfAttention in single_stream_blocks (fused QKV+MLP)."""
    if hasattr(attn, "to_qkv_mlp_proj"):
        _add_shard_spec(specs, attn.to_qkv_mlp_proj.weight, ("model", None))
        _add_shard_spec(specs, attn.to_qkv_mlp_proj.bias, ("model",))

    if hasattr(attn, "to_out") and isinstance(attn.to_out, torch.nn.Linear):
        _add_shard_spec(specs, attn.to_out.weight, (None, "model"))
        _add_shard_spec(specs, attn.to_out.bias, (None,))

    # QK-RMSNorm weights (head_dim,) must be replicated — see note in
    # _shard_flux2_joint_attention; heads are sharded, head_dim is not.
    for norm_name in ("norm_q", "norm_k"):
        norm = getattr(attn, norm_name, None)
        if norm is not None:
            _add_shard_spec(specs, getattr(norm, "weight", None), (None,))


def _shard_flux2_feed_forward(ff, specs: dict) -> None:
    _add_shard_spec(specs, ff.linear_in.weight, ("model", None))
    _add_shard_spec(specs, ff.linear_out.weight, (None, "model"))


def shard_transformer_specs(transformer) -> dict:
    """Shard Flux2Transformer2DModel (dual-stream + single-stream blocks)."""
    specs = {}

    _add_shard_spec(specs, transformer.x_embedder.weight, ("model", None))
    _add_shard_spec(specs, transformer.context_embedder.weight, ("model", None))

    # Modulation (AdaLN) projections must be REPLICATED, not column-sharded.
    # Flux2Modulation.linear emits dim*3*sets features that the block then splits
    # with torch.chunk(..., dim=-1) into shift/scale/gate, each of width `dim`.
    # Column-sharding ("model", None) splits that same last axis 8-way, so the
    # chunk boundaries (every `dim`) no longer align with the shard boundaries
    # (every dim*3*sets/8) and shift/scale/gate get misassigned across devices ->
    # the (1+scale)*x+shift modulation is applied with the wrong per-device slice.
    # (Invisible on random init where scale/shift≈0, catastrophic with trained
    # weights — output decorrelates.) These projections are tiny, so replicating
    # them costs negligible DRAM.
    for mod in (
        transformer.double_stream_modulation_img,
        transformer.double_stream_modulation_txt,
        transformer.single_stream_modulation,
    ):
        _add_shard_spec(specs, mod.linear.weight, (None, None))

    def _shard_timestep_embedding(embedder) -> None:
        if embedder is None:
            return
        _add_shard_spec(specs, embedder.linear_1.weight, ("model", None))
        _add_shard_spec(specs, embedder.linear_1.bias, ("model",))
        _add_shard_spec(specs, embedder.linear_2.weight, (None, "model"))
        _add_shard_spec(specs, embedder.linear_2.bias, (None,))

    _shard_timestep_embedding(transformer.time_guidance_embed.timestep_embedder)
    _shard_timestep_embedding(transformer.time_guidance_embed.guidance_embedder)

    for block in transformer.transformer_blocks:
        for norm_name in ("norm1", "norm1_context"):
            norm = getattr(block, norm_name, None)
            if norm is not None:
                # elementwise_affine=False → weight/bias are None; skip
                _add_shard_spec(specs, getattr(norm, "weight", None), (None,))
                _add_shard_spec(specs, getattr(norm, "bias", None), (None,))
        _shard_flux2_joint_attention(block.attn, specs)
        _shard_flux2_feed_forward(block.ff, specs)
        _shard_flux2_feed_forward(block.ff_context, specs)

    for block in transformer.single_transformer_blocks:
        if hasattr(block, "norm"):
            _add_shard_spec(specs, getattr(block.norm, "weight", None), (None,))
            _add_shard_spec(specs, getattr(block.norm, "bias", None), (None,))
        _shard_flux2_parallel_attention(block.attn, specs)

    # norm_out is AdaLayerNormContinuous: its .linear output is chunked into
    # (scale, shift), so it has the same chunk-vs-shard misalignment as the
    # modulation projections above and must be replicated.
    if hasattr(transformer.norm_out, "linear"):
        _add_shard_spec(specs, transformer.norm_out.linear.weight, (None, None))
        _add_shard_spec(specs, transformer.norm_out.linear.bias, (None,))
    _add_shard_spec(specs, transformer.proj_out.weight, (None, None))

    return specs
