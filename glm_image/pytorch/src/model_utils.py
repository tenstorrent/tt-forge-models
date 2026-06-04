# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for GLM-Image.

Model: zai-org/GLM-Image
Components:
  - text_encoder:             T5EncoderModel (ByT5-style, vocab=384, d=1472)
  - vision_language_encoder:  GlmImageForConditionalGeneration
                              (visual ViT + language model + VQ-VAE)
  - transformer:              GlmImageTransformer2DModel DiT (30 blocks)
  - vae:                      AutoencoderKL (latent_channels=16)
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "zai-org/GLM-Image"
DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Inference shape constants
#
# VAE: spatial_compression=8 (three Downsample2D / Upsample2D blocks).
# Transformer: hidden_states is the 4-D latent tensor (B, C_latent, H, W);
#   patching is done internally. The shapes below match a real forward
#   capture from GlmImageTransformer2DModel.forward (image 1024x1152,
#   latent 128x144, text seq 376, prior seq 4608).
# ---------------------------------------------------------------------------

IMAGE_H = 1024
IMAGE_W = 1152
VAE_SCALE_FACTOR = 8
LATENT_H = IMAGE_H // VAE_SCALE_FACTOR  # 128
LATENT_W = IMAGE_W // VAE_SCALE_FACTOR  # 144

NUM_CHANNELS_LATENTS = 16  # AutoencoderKL latent channels

TRANSFORMER_HIDDEN_DIM = 4096
TRANSFORMER_TIME_DIM = 512
TRANSFORMER_TEXT_SEQ = 376  # encoder_hidden_states seq dim from real forward
TRANSFORMER_TIMESTEP_VALUE = 999.0
TRANSFORMER_TARGET_SIZE = (1024.0, 1152.0)
TRANSFORMER_CROP_COORDS = (0.0, 0.0)

# T5 text encoder dims (from arch dump + real forward capture: B=16, T=38)
T5_VOCAB_SIZE = 384
T5_HIDDEN_DIM = 1472
T5_BATCH_SIZE = 16
T5_SEQ_LEN = 38

# Vision-language encoder dims (arch dump + real forward capture).
# Capture: B=1 sample with 2 images of patch grids (1,32,36) and (1,15,16) →
#   total visual patches = 32*36 + 15*16 = 1392.
# patch_size=16, temporal_patch_size=1, channels=3 → patch_pixel_dim = 768.
VL_TEXT_VOCAB_SIZE = 168064
VL_TEXT_HIDDEN_DIM = 4096
VL_VISION_HIDDEN_DIM = 1536
VL_BATCH_SIZE = 1
VL_TEXT_SEQ = 394
VL_PATCH_PIXEL_DIM = 3 * 16 * 16  # channels * patch_h * patch_w
VL_IMAGE_GRID_THW = ((1, 32, 36), (1, 15, 16))
VL_IMAGES_PER_SAMPLE = (2,)
VL_LOGITS_TO_KEEP = 1

# Prior token vocabulary + sequence length used by GlmImageTransformer2DModel
PRIOR_VOCAB_SIZE = 16384
PRIOR_SEQ_LEN = 4608

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Load T5 text encoder from the text_encoder subfolder."""
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vision_language_encoder(dtype: torch.dtype = DTYPE):
    """Load GlmImageForConditionalGeneration from the vision_language_encoder subfolder."""
    from transformers import GlmImageForConditionalGeneration

    return GlmImageForConditionalGeneration.from_pretrained(
        REPO_ID,
        subfolder="vision_language_encoder",
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load GlmImageTransformer2DModel from the transformer subfolder."""
    from diffusers import GlmImageTransformer2DModel

    return GlmImageTransformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """Load AutoencoderKL from the vae subfolder."""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Component input builders
# ---------------------------------------------------------------------------


def load_text_encoder_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for the T5 text encoder.

    Shapes match a real forward capture:
      input_ids      (16, 38) int64
      attention_mask (16, 38) int64
    """
    input_ids = torch.randint(
        0, T5_VOCAB_SIZE, (T5_BATCH_SIZE, T5_SEQ_LEN), dtype=torch.long
    )
    attention_mask = torch.ones(T5_BATCH_SIZE, T5_SEQ_LEN, dtype=torch.long)
    return [input_ids, attention_mask]


def load_vision_language_encoder_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for GlmImageForConditionalGeneration.

    Shapes match a real forward capture:
      input_ids          (1, 394)            int64
      attention_mask     (1, 394)            int64
      image_grid_thw     [[1,32,36],[1,15,16]] int64
      images_per_sample  [2]                 int64
      cache_position     arange(394)         int64
      logits_to_keep     scalar tensor 1     int64
    """
    input_ids = torch.randint(
        0, VL_TEXT_VOCAB_SIZE, (VL_BATCH_SIZE, VL_TEXT_SEQ), dtype=torch.long
    )
    attention_mask = torch.ones(VL_BATCH_SIZE, VL_TEXT_SEQ, dtype=torch.long)
    image_grid_thw = torch.tensor(list(VL_IMAGE_GRID_THW), dtype=torch.long)
    images_per_sample = torch.tensor(list(VL_IMAGES_PER_SAMPLE), dtype=torch.long)
    cache_position = torch.arange(VL_TEXT_SEQ, dtype=torch.long)
    logits_to_keep = torch.tensor(VL_LOGITS_TO_KEEP, dtype=torch.long)
    return [
        input_ids,
        attention_mask,
        image_grid_thw,
        images_per_sample,
        cache_position,
        logits_to_keep,
    ]


def load_transformer_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for GlmImageTransformer2DModel.

    Shapes match a real forward capture:
      hidden_states         (1, 16, 128, 144)
      encoder_hidden_states (1, 376, 1472)
      prior_token_id        (1, 4608) int64
      prior_token_drop      (1, 4608) bool
      timestep              tensor([999.])
      target_size           tensor([[1024., 1152.]]) bfloat16
      crop_coords           tensor([[0., 0.]])       bfloat16
    """
    hidden_states = torch.randn(
        1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W, dtype=dtype
    )
    encoder_hidden_states = torch.randn(
        1, TRANSFORMER_TEXT_SEQ, T5_HIDDEN_DIM, dtype=dtype
    )
    gen = torch.Generator().manual_seed(0)
    prior_token_id = torch.randint(
        0, PRIOR_VOCAB_SIZE, (1, PRIOR_SEQ_LEN), dtype=torch.long, generator=gen
    )
    prior_token_drop = torch.randint(
        0, PRIOR_VOCAB_SIZE, (1, PRIOR_SEQ_LEN), dtype=torch.long
    )
    timestep = torch.tensor([TRANSFORMER_TIMESTEP_VALUE])
    target_size = torch.tensor([list(TRANSFORMER_TARGET_SIZE)], dtype=torch.bfloat16)
    crop_coords = torch.tensor([list(TRANSFORMER_CROP_COORDS)], dtype=torch.bfloat16)
    return [
        hidden_states,
        encoder_hidden_states,
        prior_token_id,
        prior_token_drop,
        timestep,
        target_size,
        crop_coords,
    ]


def load_vae_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic latent input for VAEDecoderWrapper.

    Shape matches a real AutoencoderKL.decode capture:
      z (1, 16, 128, 144)
    """
    z = torch.randn(
        1,
        NUM_CHANNELS_LATENTS,
        LATENT_H,
        LATENT_W,
        dtype=dtype,
    )
    return [z]


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only AutoencoderKL decoder as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class GlmImageTransformerWrapper(torch.nn.Module):
    """Simplify GlmImageTransformer2DModel forward to tensor-only inputs/outputs."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        prior_token_id,
        prior_token_drop,
        timestep,
        target_size,
        crop_coords,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            prior_token_id=prior_token_id,
            prior_token_drop=prior_token_drop,
            timestep=timestep,
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
        )[0]


class GlmImageVisionLanguageWrapper(torch.nn.Module):
    """Simplify GlmImageForConditionalGeneration forward to tensor-only inputs."""

    def __init__(self, vlm):
        super().__init__()
        self.vlm = vlm

    def forward(
        self,
        input_ids,
        attention_mask,
        image_grid_thw,
        images_per_sample,
        cache_position,
        logits_to_keep,
    ):
        return self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            images_per_sample=images_per_sample,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        ).logits


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_text_encoder_specs(encoder) -> dict:
    """Shard specs for T5 text encoder.

    Column-parallel (q, k, v, wi_0, wi_1): ("model", "batch")
    Row-parallel   (o, wo):                ("batch", "model")
    """
    specs = {}

    if hasattr(encoder, "shared"):
        specs[encoder.shared.weight] = (None, "batch")

    stack = getattr(encoder, "encoder", None)
    if stack is None:
        return specs

    if hasattr(stack, "embed_tokens"):
        specs[stack.embed_tokens.weight] = (None, "batch")

    blocks = getattr(stack, "block", None)
    if blocks is None:
        return specs

    for block in blocks:
        self_attn_layer = block.layer[0]
        attn = self_attn_layer.SelfAttention
        for proj_name in ("q", "k", "v"):
            proj = getattr(attn, proj_name)
            specs[proj.weight] = ("model", "batch")
            if getattr(proj, "bias", None) is not None:
                specs[proj.bias] = ("model",)
        specs[attn.o.weight] = ("batch", "model")
        if getattr(attn.o, "bias", None) is not None:
            specs[attn.o.bias] = ("batch",)
        if hasattr(self_attn_layer, "layer_norm"):
            specs[self_attn_layer.layer_norm.weight] = ("batch",)

        ff_layer = block.layer[1]
        dense = ff_layer.DenseReluDense
        for proj_name in ("wi_0", "wi_1"):
            if hasattr(dense, proj_name):
                proj = getattr(dense, proj_name)
                specs[proj.weight] = ("model", "batch")
                if getattr(proj, "bias", None) is not None:
                    specs[proj.bias] = ("model",)
        if hasattr(dense, "wo"):
            specs[dense.wo.weight] = ("batch", "model")
            if getattr(dense.wo, "bias", None) is not None:
                specs[dense.wo.bias] = ("batch",)
        if hasattr(ff_layer, "layer_norm"):
            specs[ff_layer.layer_norm.weight] = ("batch",)

    if hasattr(stack, "final_layer_norm"):
        specs[stack.final_layer_norm.weight] = ("batch",)

    return specs


def shard_vision_language_encoder_specs(vlm) -> dict:
    """Shard specs for GlmImageForConditionalGeneration.

    Column-parallel (qkv, q, k, v, gate_up, fc1): ("model", "batch")
    Row-parallel   (proj/o, down_proj, fc2):       ("batch", "model")
    Replicated norms / embeddings as required.
    """
    specs = {}

    model = getattr(vlm, "model", vlm)

    # Visual tower (ViT-style)
    visual = getattr(model, "visual", None)
    if visual is not None:
        if hasattr(visual, "embeddings") and hasattr(
            visual.embeddings, "position_embedding"
        ):
            specs[visual.embeddings.position_embedding.weight] = (None, "batch")
        if hasattr(visual, "patch_embed") and hasattr(visual.patch_embed, "proj"):
            specs[visual.patch_embed.proj.weight] = ("batch", None, None, None)
            if visual.patch_embed.proj.bias is not None:
                specs[visual.patch_embed.proj.bias] = ("batch",)

        for block in getattr(visual, "blocks", []) or []:
            specs[block.norm1.weight] = ("batch",)
            if block.norm1.bias is not None:
                specs[block.norm1.bias] = ("batch",)
            specs[block.norm2.weight] = ("batch",)
            if block.norm2.bias is not None:
                specs[block.norm2.bias] = ("batch",)

            attn = block.attn
            specs[attn.qkv.weight] = ("model", "batch")
            if attn.qkv.bias is not None:
                specs[attn.qkv.bias] = ("model",)
            specs[attn.proj.weight] = ("batch", "model")
            if attn.proj.bias is not None:
                specs[attn.proj.bias] = ("batch",)

            mlp = block.mlp
            specs[mlp.fc1.weight] = ("model", "batch")
            if mlp.fc1.bias is not None:
                specs[mlp.fc1.bias] = ("model",)
            specs[mlp.fc2.weight] = ("batch", "model")
            if mlp.fc2.bias is not None:
                specs[mlp.fc2.bias] = ("batch",)

    # Language model (decoder-only)
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        if hasattr(language_model, "embed_tokens"):
            specs[language_model.embed_tokens.weight] = (None, "batch")

        for layer in getattr(language_model, "layers", []) or []:
            sa = layer.self_attn
            specs[sa.q_proj.weight] = ("model", "batch")
            if sa.q_proj.bias is not None:
                specs[sa.q_proj.bias] = ("model",)
            specs[sa.k_proj.weight] = ("model", "batch")
            if sa.k_proj.bias is not None:
                specs[sa.k_proj.bias] = ("model",)
            specs[sa.v_proj.weight] = ("model", "batch")
            if sa.v_proj.bias is not None:
                specs[sa.v_proj.bias] = ("model",)
            specs[sa.o_proj.weight] = ("batch", "model")

            mlp = layer.mlp
            specs[mlp.gate_up_proj.weight] = ("model", "batch")
            specs[mlp.down_proj.weight] = ("batch", "model")

            specs[layer.input_layernorm.weight] = ("batch",)
            specs[layer.post_attention_layernorm.weight] = ("batch",)
            if hasattr(layer, "post_self_attn_layernorm"):
                specs[layer.post_self_attn_layernorm.weight] = ("batch",)
            if hasattr(layer, "post_mlp_layernorm"):
                specs[layer.post_mlp_layernorm.weight] = ("batch",)

        if hasattr(language_model, "norm"):
            specs[language_model.norm.weight] = ("batch",)

    # VQ-VAE codebook + quant convs.
    # quant_conv (1x1 Conv2d) is column-parallel (shard Cout) and
    # post_quant_conv (1x1 Conv2d) is row-parallel (shard Cin) — channel-only
    # sharding, matching the rule that spatial dims must stay replicated
    # because halo exchange isn't supported.
    vqmodel = getattr(model, "vqmodel", None)
    if vqmodel is not None:
        if hasattr(vqmodel, "quantize") and hasattr(vqmodel.quantize, "embedding"):
            specs[vqmodel.quantize.embedding.weight] = (None, "batch")
        if hasattr(vqmodel, "quant_conv"):
            specs[vqmodel.quant_conv.weight] = ("model", None, None, None)
            if vqmodel.quant_conv.bias is not None:
                specs[vqmodel.quant_conv.bias] = ("model",)
        if hasattr(vqmodel, "post_quant_conv"):
            specs[vqmodel.post_quant_conv.weight] = (None, "model", None, None)
            if vqmodel.post_quant_conv.bias is not None:
                specs[vqmodel.post_quant_conv.bias] = (
                    None,
                )  # replicated post-all_reduce

    if hasattr(vlm, "lm_head"):
        specs[vlm.lm_head.weight] = ("model", "batch")

    return specs


def _shard_resnet_block_2d(block, specs: dict) -> None:
    """Megatron-style sharding for AutoencoderKL ResnetBlock2D (Conv2d, NCHW).

    The two consecutive convs in a ResnetBlock are sharded as a Megatron pair
    so only ONE all_reduce is needed per block, and the intermediate tensor
    (after norm + SiLU) stays partitioned along Cout.

      conv1 (column-parallel, shard Cout = weight dim 0):
        weight ("model", None, None, None)   # split Cout across "model"
        bias   ("model",)                    # matches Cout
        # input is replicated, output is sharded along Cout — no comm needed.

      conv2 (row-parallel, shard Cin = weight dim 1):
        weight (None, "model", None, None)   # split Cin across "model"
        bias   (None,)                       # MUST be replicated:
                                             # output is replicated after the
                                             # all_reduce; a sharded bias would
                                             # be summed N times by the reduce.
        # input is the conv1 output (already Cout-sharded → matches Cin here),
        # so no all_gather between conv1 → conv2. One all_reduce at the end
        # produces the fully-replicated block output.

      conv_shortcut (replicated):
        weight (None, None, None, None)
        bias   (None,)
        # conv2 produces a replicated output; the residual add
        # `conv2_out + shortcut_out` requires the shortcut to also be
        # replicated, so we don't shard it.

    Spatial dims (kH, kW) are NEVER sharded — that would require a halo
    exchange (neighbor_pad / slice_reshard) collective which TTIR/TTNN MLIR
    does not currently expose. Channel sharding is the only viable strategy.
    """
    if hasattr(block, "norm1"):
        specs[block.norm1.weight] = ("batch",)
        if block.norm1.bias is not None:
            specs[block.norm1.bias] = ("batch",)

    specs[block.conv1.weight] = ("model", None, None, None)
    if block.conv1.bias is not None:
        specs[block.conv1.bias] = ("model",)

    if hasattr(block, "norm2"):
        specs[block.norm2.weight] = ("batch",)
        if block.norm2.bias is not None:
            specs[block.norm2.bias] = ("batch",)

    specs[block.conv2.weight] = (None, "model", None, None)
    if block.conv2.bias is not None:
        specs[block.conv2.bias] = (None,)

    if getattr(block, "conv_shortcut", None) is not None:
        specs[block.conv_shortcut.weight] = (None, None, None, None)
        if block.conv_shortcut.bias is not None:
            specs[block.conv_shortcut.bias] = (None,)


def shard_vae_specs(vae) -> dict:
    """Shard specs for AutoencoderKL (decoder-only path).

    For Conv2d weight (Cout, Cin, kH, kW) the only viable sharding axes are
    the two channel dims:
      * Cout (weight dim 0) → column-parallel: no comm in, partitioned output.
      * Cin  (weight dim 1) → row-parallel:   partitioned input, all_reduce out.

    Spatial sharding (kH/kW) is intentionally avoided — it would require a
    halo exchange collective (each tile needs (k-1)/2 elements from each
    neighbour), which TTIR/TTNN MLIR does not currently expose. Batch
    sharding is also unused here because the decoder runs with B=1 in this
    pipeline; per-block work is parallelised across the channel dim instead.

    Mapping (mesh axis "model" carries the channel split):
      Column-parallel (entry conv_in, resnet conv1, upsampler conv):
        weight ("model", None, None, None),  bias ("model",)
      Row-parallel   (resnet conv2, exit conv_out):
        weight (None, "model", None, None),  bias (None,)
        # bias replicated because the row-parallel output is replicated
        # after the all_reduce — a sharded bias would be summed N times.
      Replicated     (conv_shortcut):
        weight (None, None, None, None),     bias (None,)
        # must match the post-all_reduce replicated state of conv2 so the
        # residual add doesn't need a re-shard.

    Norm / mid-block attention specs are replicated along the "batch" axis
    only, mirroring the conventions used elsewhere in the pipeline.
    """
    specs = {}

    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        return specs

    if hasattr(decoder, "conv_in"):
        specs[decoder.conv_in.weight] = ("model", None, None, None)
        if decoder.conv_in.bias is not None:
            specs[decoder.conv_in.bias] = ("model",)

    mid_block = getattr(decoder, "mid_block", None)
    if mid_block is not None:
        for resnet in getattr(mid_block, "resnets", []) or []:
            _shard_resnet_block_2d(resnet, specs)
        for attn in getattr(mid_block, "attentions", []) or []:
            if attn is None:
                continue
            if hasattr(attn, "group_norm") and attn.group_norm is not None:
                specs[attn.group_norm.weight] = ("batch",)
                if attn.group_norm.bias is not None:
                    specs[attn.group_norm.bias] = ("batch",)
            for proj_name in ("to_q", "to_k", "to_v"):
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    specs[proj.weight] = ("model", "batch")
                    if proj.bias is not None:
                        specs[proj.bias] = ("model",)
            if hasattr(attn, "to_out"):
                out = attn.to_out
                target = (
                    out[0]
                    if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                    else out
                )
                specs[target.weight] = ("batch", "model")
                if target.bias is not None:
                    specs[target.bias] = ("batch",)

    for up_block in getattr(decoder, "up_blocks", []) or []:
        for resnet in getattr(up_block, "resnets", []) or []:
            _shard_resnet_block_2d(resnet, specs)
        for upsampler in getattr(up_block, "upsamplers", []) or []:
            specs[upsampler.conv.weight] = ("model", None, None, None)
            if upsampler.conv.bias is not None:
                specs[upsampler.conv.bias] = ("model",)

    if hasattr(decoder, "conv_norm_out"):
        specs[decoder.conv_norm_out.weight] = ("batch",)
        if decoder.conv_norm_out.bias is not None:
            specs[decoder.conv_norm_out.bias] = ("batch",)

    if hasattr(decoder, "conv_out"):
        specs[decoder.conv_out.weight] = (None, "model", None, None)
        if decoder.conv_out.bias is not None:
            specs[decoder.conv_out.bias] = (None,)

    return specs


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for GlmImageTransformer2DModel.

    Column-parallel (Q, K, V, FFN up, modulation linear): ("model", "batch")
    Row-parallel   (O, FFN down, proj_out):               ("batch", "model")
    """
    specs = {}

    # Image / glyph / prior projectors are entry-point linears: shard Cout.
    if hasattr(transformer, "image_projector") and hasattr(
        transformer.image_projector, "proj"
    ):
        proj = transformer.image_projector.proj
        specs[proj.weight] = ("model", "batch")
        if proj.bias is not None:
            specs[proj.bias] = ("model",)

    if hasattr(transformer, "glyph_projector"):
        gp = transformer.glyph_projector
        if hasattr(gp, "net"):
            if hasattr(gp.net[0], "proj"):
                specs[gp.net[0].proj.weight] = ("model", "batch")
                if gp.net[0].proj.bias is not None:
                    specs[gp.net[0].proj.bias] = ("model",)
            specs[gp.net[2].weight] = ("batch", "model")
            if gp.net[2].bias is not None:
                specs[gp.net[2].bias] = ("batch",)

    if hasattr(transformer, "prior_token_embedding"):
        specs[transformer.prior_token_embedding.weight] = (None, "batch")

    if hasattr(transformer, "prior_projector"):
        pp = transformer.prior_projector
        if hasattr(pp, "net"):
            if hasattr(pp.net[0], "proj"):
                specs[pp.net[0].proj.weight] = ("model", "batch")
                if pp.net[0].proj.bias is not None:
                    specs[pp.net[0].proj.bias] = ("model",)
            specs[pp.net[2].weight] = ("batch", "model")
            if pp.net[2].bias is not None:
                specs[pp.net[2].bias] = ("batch",)

    # Time / size conditioning embedders (small replicated linears).
    if hasattr(transformer, "time_condition_embed"):
        tce = transformer.time_condition_embed
        for sub in ("timestep_embedder", "condition_embedder"):
            mod = getattr(tce, sub, None)
            if mod is None:
                continue
            for lin_name in ("linear_1", "linear_2"):
                if hasattr(mod, lin_name):
                    lin = getattr(mod, lin_name)
                    specs[lin.weight] = ("batch", None)
                    if lin.bias is not None:
                        specs[lin.bias] = ("batch",)

    for block in getattr(transformer, "transformer_blocks", []) or []:
        for norm_name in ("norm1", "norm1_context"):
            mod = getattr(block, norm_name, None)
            if mod is not None and hasattr(mod, "linear"):
                lin = mod.linear
                specs[lin.weight] = ("model", "batch")
                if lin.bias is not None:
                    specs[lin.bias] = ("model",)

        if hasattr(block, "attn1"):
            attn = block.attn1
            for proj_name in ("to_q", "to_k", "to_v"):
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    specs[proj.weight] = ("model", "batch")
                    if proj.bias is not None:
                        specs[proj.bias] = ("model",)
            if hasattr(attn, "to_out"):
                out = attn.to_out
                target = (
                    out[0]
                    if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                    else out
                )
                specs[target.weight] = ("batch", "model")
                if target.bias is not None:
                    specs[target.bias] = ("batch",)

        if hasattr(block, "ff"):
            ff = block.ff
            if hasattr(ff.net[0], "proj"):
                specs[ff.net[0].proj.weight] = ("model", "batch")
                if ff.net[0].proj.bias is not None:
                    specs[ff.net[0].proj.bias] = ("model",)
            specs[ff.net[2].weight] = ("batch", "model")
            if ff.net[2].bias is not None:
                specs[ff.net[2].bias] = ("batch",)

    if hasattr(transformer, "norm_out") and hasattr(transformer.norm_out, "linear"):
        lin = transformer.norm_out.linear
        specs[lin.weight] = ("model", "batch")
        if lin.bias is not None:
            specs[lin.bias] = ("model",)

    if hasattr(transformer, "proj_out"):
        specs[transformer.proj_out.weight] = (None, "batch")
        if transformer.proj_out.bias is not None:
            specs[transformer.proj_out.bias] = (None,)

    return specs
