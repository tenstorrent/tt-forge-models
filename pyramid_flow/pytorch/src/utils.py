# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Component loaders for the Pyramid Flow SD3 video-generation pipeline.

`rain1011/pyramid-flow-sd3` is a text-to-video flow-matching model with the
Stable-Diffusion-3 component layout:

  * text_encoder    -> CLIP-L  (CLIPTextModelWithProjection)
  * text_encoder_2  -> CLIP-G  (CLIPTextModelWithProjection)
  * text_encoder_3  -> T5-XXL  (T5EncoderModel)
  * diffusion_transformer_768p -> PyramidDiffusionMMDiT (SD3 MMDiT denoiser)
  * causal_video_vae           -> CausalVideoVAE

The MMDiT has no diffusers integration, so its model code is vendored under
`mmdit_modules/` (verbatim from upstream, with the `trainer_misc`
sequence-parallel imports replaced by a local stub). Real pretrained weights
are loaded for every component.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .mmdit_modules import PyramidDiffusionMMDiT

REPO_ID = "rain1011/pyramid-flow-sd3"

# ============================================================================
# Native 768p generation geometry (from the model card / pipeline defaults)
#   video 1280x768, VAE downsample 8 -> latent 160x96, 16 latent channels.
#   frame_per_unit=1, so the first frame is a real single-frame (T=1) denoise
#   step at the highest pyramid stage -> native spatial resolution.
# ============================================================================
_LATENT_CHANNELS = 16
_LATENT_T = 1
_LATENT_H = 96   # 768 // 8
_LATENT_W = 160  # 1280 // 8
_TEXT_SEQ_LEN = 128
_JOINT_ATTENTION_DIM = 4096   # T5-XXL hidden, into context_embedder
_POOLED_PROJECTION_DIM = 2048  # CLIP-L (768) + CLIP-G (1280) pooled


# ============================================================================
# Denoiser (MMDiT) -- the heavy compute target; must run on device
# ============================================================================


class PyramidMMDiTWrapper(nn.Module):
    """Device-friendly single-step wrapper around PyramidDiffusionMMDiT.

    The upstream ``forward`` calls ``merge_input``, which builds the temporal
    RoPE embedding and the (T5 + temporal-causal) attention mask inside the
    forward pass with ``torch.arange``/``torch.zeros``/index assignments. Those
    tensors depend only on the input **shapes** (static), but tracing them into
    the compiled graph produces host (CPU) constants that conflict with the
    on-device tensors ("Input tensor is not an XLA tensor: CPUFloatType").

    We hoist that static glue out of the graph: ``merge_input`` is run once on
    the host at construction time for the fixed single-stage geometry, and the
    resulting RoPE embedding + attention mask are stored as buffers. ``forward``
    then runs only the device-worthy compute — conditioning embedders, the 3D
    patch-embed conv, the MMDiT blocks, output norm/projection, and the
    reshape-only un-patchify — consuming the precomputed buffers.
    """

    def __init__(self, model: PyramidDiffusionMMDiT, sample_shape, text_seq_len):
        super().__init__()
        self.model = model

        b, c, t, h, w = sample_shape
        dummy = torch.zeros(b, c, t, h, w, dtype=next(model.parameters()).dtype)
        enc_mask = torch.ones(b, text_seq_len, dtype=torch.long)
        with torch.no_grad():
            (
                _hidden,
                hidden_length,
                temps,
                heights,
                widths,
                trainable_token_list,
                _enc_attn,
                attention_mask,
                image_rotary_emb,
            ) = model.merge_input(
                [[dummy]], text_seq_len, enc_mask
            )

        # Static, shape-derived integer metadata (kept as Python lists).
        self._hidden_length = list(hidden_length)
        self._temps = list(temps)
        self._heights = list(heights)
        self._widths = list(widths)
        self._trainable_token_list = list(trainable_token_list)

        # Single stage -> one mask / one RoPE tensor; store as buffers so the
        # harness moves them to device. The bool mask is left untouched by a
        # later float `.to(dtype)` (only floating buffers are cast).
        self.register_buffer("attn_mask", attention_mask[0], persistent=False)
        self.register_buffer("rope_emb", image_rotary_emb[0], persistent=False)

    def forward(
        self,
        latent,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        timestep_ratio,
    ):
        m = self.model

        temb = m.time_text_embed(timestep_ratio, pooled_projections)
        encoder_hidden_states = m.context_embedder(encoder_hidden_states)

        # 3D patch-embed conv (device work); single stage -> single tensor.
        hidden_states = m.pos_embed([[latent]])
        hidden_states = torch.cat(hidden_states, dim=1)

        attention_mask = [self.attn_mask]
        image_rotary_emb = [self.rope_emb]

        for block in m.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                temb=temb,
                attention_mask=attention_mask,
                hidden_length=self._hidden_length,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = m.norm_out(hidden_states, temb, hidden_length=self._hidden_length)
        hidden_states = m.proj_out(hidden_states)

        output = m.split_output(
            hidden_states,
            self._hidden_length,
            self._temps,
            self._heights,
            self._widths,
            self._trainable_token_list,
        )
        return output[0]


def load_transformer(dtype: torch.dtype) -> PyramidMMDiTWrapper:
    """Load the SD3 MMDiT denoiser (768p) with real pretrained weights.

    The extra kwargs match upstream `build_pyramid_dit(model_name=
    'pyramid_mmdit', ...)`: they are not stored in config.json but are required
    for the inference forward path (temporal RoPE + T5 mask, no flash attn).
    """
    model = PyramidDiffusionMMDiT.from_pretrained(
        REPO_ID,
        subfolder="diffusion_transformer_768p",
        use_flash_attn=False,
        use_t5_mask=True,
        add_temp_pos_embed=True,
        temp_pos_embed_type="rope",
        use_temporal_causal=True,
        interp_condition_pos=True,
        use_gradient_checkpointing=False,
    )
    model = model.to(dtype=dtype).eval()
    sample_shape = (1, _LATENT_CHANNELS, _LATENT_T, _LATENT_H, _LATENT_W)
    return PyramidMMDiTWrapper(model, sample_shape, _TEXT_SEQ_LEN).eval()


def load_transformer_inputs(dtype: torch.dtype) -> Dict[str, Any]:
    """Build a single native-resolution denoising-step input for the MMDiT.

    The latent is a plain top-level tensor `[B, C_latent, T, H, W]` at the
    native 768p latent geometry (first frame, T=1); `PyramidMMDiTWrapper`
    rebuilds the single-stage nested `sample` list internally.
    """
    batch_size = 1
    latent = torch.randn(
        batch_size,
        _LATENT_CHANNELS,
        _LATENT_T,
        _LATENT_H,
        _LATENT_W,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        batch_size, _TEXT_SEQ_LEN, _JOINT_ATTENTION_DIM, dtype=dtype
    )
    encoder_attention_mask = torch.ones(batch_size, _TEXT_SEQ_LEN, dtype=torch.long)
    pooled_projections = torch.randn(
        batch_size, _POOLED_PROJECTION_DIM, dtype=dtype
    )
    timestep_ratio = torch.tensor([500.0] * batch_size, dtype=dtype)

    return {
        "latent": latent,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "pooled_projections": pooled_projections,
        "timestep_ratio": timestep_ratio,
    }


# ============================================================================
# Text encoders (standard transformers) -- CLIP-L, CLIP-G, T5-XXL
# ============================================================================


def load_text_encoder(dtype: torch.dtype):
    """CLIP-L text encoder (CLIPTextModelWithProjection)."""
    from transformers import CLIPTextModelWithProjection

    model = CLIPTextModelWithProjection.from_pretrained(
        REPO_ID, subfolder="text_encoder", torch_dtype=dtype
    )
    return model.eval()


def load_text_encoder_2(dtype: torch.dtype):
    """CLIP-G text encoder (CLIPTextModelWithProjection)."""
    from transformers import CLIPTextModelWithProjection

    model = CLIPTextModelWithProjection.from_pretrained(
        REPO_ID, subfolder="text_encoder_2", torch_dtype=dtype
    )
    return model.eval()


def load_text_encoder_3(dtype: torch.dtype):
    """T5-XXL text encoder (T5EncoderModel)."""
    from transformers import T5EncoderModel

    model = T5EncoderModel.from_pretrained(
        REPO_ID, subfolder="text_encoder_3", torch_dtype=dtype
    )
    return model.eval()


def _clip_inputs() -> Dict[str, Any]:
    # CLIP text models use a 77-token context window.
    input_ids = torch.randint(0, 49407, (1, 77), dtype=torch.long)
    return {"input_ids": input_ids}


def load_text_encoder_inputs(dtype: torch.dtype) -> Dict[str, Any]:
    return _clip_inputs()


def load_text_encoder_2_inputs(dtype: torch.dtype) -> Dict[str, Any]:
    return _clip_inputs()


def load_text_encoder_3_inputs(dtype: torch.dtype) -> Dict[str, Any]:
    # T5-XXL encoder; SD3 caps T5 at 128 tokens.
    input_ids = torch.randint(0, 32128, (1, _TEXT_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, _TEXT_SEQ_LEN, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
