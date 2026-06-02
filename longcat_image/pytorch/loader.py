# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Image model loader implementation for text-to-image generation.

LongCat-Image is a flow-matching DiT from Meituan (`LongCatImagePipeline`,
stock diffusers >= 0.37). Repository:
    https://huggingface.co/meituan-longcat/LongCat-Image

Pipeline components and bringup coverage:
  - text_encoder (Qwen2_5_VLForConditionalGeneration, ~8.3 B) — covered by
    the existing `qwen_2_5_vl` family.
  - vae (AutoencoderKL, ~84 M) — same arch as FLUX VAE; covered by `flux`.
  - transformer (LongCatImageTransformer2DModel, ~6.3 B) — the novel
    component this loader scaffolds.

`load_model()` returns only the diffusion transformer; `load_inputs()`
synthesizes the kwargs matching the pipeline call site (packed latents,
encoder_hidden_states from text encoder, txt_ids/img_ids position grids,
flow-matching timestep in (0, 1)).

**Random-weights mode (first-pass compile).** No safetensors are
downloaded. The transformer is instantiated from its HF `config.json`
with random init, which is enough to exercise the compile path and
surface MLIR / runtime issues. PCC validation against real weights is a
follow-up phase.
"""
import json
from typing import Optional

import torch
from diffusers import LongCatImageTransformer2DModel
from diffusers.models.transformers.transformer_longcat_image import (
    LongCatImageAttention,
    _get_qkv_projections,
    apply_rotary_emb,
)
from huggingface_hub import hf_hub_download

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


_HF_REPO = "meituan-longcat/LongCat-Image"


class LongCatImageManualAttnProcessor:
    """Workaround for tt-xla MHLO -> StableHLO bug triggered by SDPA.

    `nn.functional.scaled_dot_product_attention` decomposes into a path that
    inserts a `.any(dim=-1)` row-validity reduction (an `AnyComputation`
    reducer whose body contains a `select` over local constants). torch-xla's
    MHLO -> StableHLO bridge mis-translates that body, producing an
    `'stablehlo.select' op using value defined outside the region` error at
    module verification. See `feedback_sdpa_stablehlo.md` (BAGEL VAE was the
    first incident).

    This processor mirrors `LongCatImageAttnProcessor` exactly except it
    replaces `dispatch_attention_fn(...)` with explicit matmul-softmax-matmul.
    Numerically equivalent on the no-mask / no-dropout path that
    LongCat-Image actually uses (the pipeline never passes `attention_mask`
    to the transformer).

    Installed by `ModelLoader.load_model()`. The long-term fix belongs in
    tt-mlir / torch-xla, not the model.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        pass

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        (
            query,
            key,
            value,
            encoder_query,
            encoder_key,
            encoder_value,
        ) = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        # (B, S, H*D) -> (B, S, H, D)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Explicit matmul-softmax-matmul. (B, S, H, D) -> (B, H, S, D).
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        scale = q.shape[-1] ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)
        out = torch.matmul(attn_weights, v)
        hidden_states = out.transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states


class ModelVariant(StrEnum):
    """Available LongCat-Image variants. One pytest test per variant."""

    IMAGE_512PX = "Image_512PX"


class ModelLoader(ForgeModel):
    """LongCat-Image diffusion transformer loader (random-init, no weight download)."""

    _VARIANTS = {
        ModelVariant.IMAGE_512PX: ModelConfig(pretrained_model_name=_HF_REPO),
    }
    DEFAULT_VARIANT = ModelVariant.IMAGE_512PX

    # Pipeline-side constants used by load_inputs(). Derived from the
    # pipeline source and the transformer config.json at _HF_REPO:
    #   transformer.in_channels        = 64   (packed latents: 16 ch * 2x2 pack)
    #   transformer.joint_attention_dim = 3584 (Qwen2.5-VL hidden_size)
    #   pipeline.tokenizer_max_length   = 512
    #   vae_scale_factor                = 8   (FLUX-style AutoencoderKL)
    #
    # 512px target image → latent 64x64 (vae_scale_factor=8) → packed grid
    # 32x32 → 1024 patches per pipeline `_pack_latents`. Chosen as a middle
    # ground between the canonical 1024px (4096 patches) and a minimal
    # probe; keeps compile workload similar to the sister `sana` bringup.
    _IMAGE_HW = 512
    _VAE_SCALE_FACTOR = 8
    _PACK = 2
    _LATENT_CHANNELS = 16
    _IN_CHANNELS = 64  # _LATENT_CHANNELS * _PACK * _PACK
    _MAX_SEQUENCE_LENGTH = 512
    _JOINT_ATTENTION_DIM = 3584
    _ROPE_AXES = 3  # axes_dims_rope length

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LongCat-Image",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the LongCatImageTransformer2DModel with random init.

        Pulls only the small `transformer/config.json` from the HF cache;
        no safetensors are downloaded. The model is constructed with
        random weights (config-driven `from_config`), which is sufficient
        for compile-path validation. PCC against pretrained weights is a
        separate phase.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        cfg = json.load(open(hf_hub_download(_HF_REPO, "transformer/config.json")))
        # Strip diffusers bookkeeping keys that from_config rejects/warns on.
        for k in ("_class_name", "_diffusers_version", "_name_or_path"):
            cfg.pop(k, None)

        model = LongCatImageTransformer2DModel.from_config(cfg)

        # Swap every attention module's processor to the manual matmul-softmax-
        # matmul variant. The default LongCatImageAttnProcessor routes through
        # F.scaled_dot_product_attention whose XLA decomposition produces a
        # StableHLO-illegal AnyComputation reducer (see processor docstring).
        for m in model.modules():
            if isinstance(m, LongCatImageAttention):
                m.set_processor(LongCatImageManualAttnProcessor())

        model = model.to(dtype=dtype)
        model.eval()
        self._model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Synthesize transformer inputs matching the LongCatImagePipeline call site.

        Pipeline invokes the transformer as::

            self.transformer(
                hidden_states=latents,          # (B, N_patches, in_channels=64)
                timestep=timestep / 1000,       # (B,) float in (0, 1)
                guidance=None,                  # guidance_embeds=False
                encoder_hidden_states=prompt_embeds,   # (B, S_text, joint_dim=3584)
                txt_ids=text_ids,               # (S_text, 3)   -- not batched
                img_ids=latent_image_ids,       # (N_patches, 3) -- not batched
                return_dict=False,
            )
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Packed-grid side length: image / vae_scale / pack.
        packed_hw = self._IMAGE_HW // (self._VAE_SCALE_FACTOR * self._PACK)
        num_patches = packed_hw * packed_hw

        hidden_states = torch.randn(
            batch_size, num_patches, self._IN_CHANNELS, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            self._MAX_SEQUENCE_LENGTH,
            self._JOINT_ATTENTION_DIM,
            dtype=dtype,
        )
        # Flow-matching timestep is a float in (0, 1) (pipeline divides by 1000
        # before passing in; here we synthesize the post-divide value).
        timestep = torch.tensor([1.0] * batch_size, dtype=dtype)

        # Position grids are float32 in the pipeline (not bf16); the model
        # casts internally via pos_embed. Shapes match `prepare_pos_ids`:
        # (num_token, 3) for text, (H*W, 3) for image.
        txt_ids = torch.zeros(self._MAX_SEQUENCE_LENGTH, self._ROPE_AXES)
        img_ids = torch.zeros(num_patches, self._ROPE_AXES)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            "return_dict": False,
        }

    def unpack_forward_output(self, output):
        """Transformer returns a 1-tuple under return_dict=False, or a
        Transformer2DModelOutput with `.sample` otherwise."""
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
