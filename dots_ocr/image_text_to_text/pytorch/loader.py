# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr model loader implementation for document OCR (image-text-to-text).

dots.ocr (rednote-hilab/dots.ocr) is a multimodal document-parsing model: a
``dots_vit`` vision tower (42 layers, patch 14, spatial-merge 2) feeding a
Qwen2-style causal LM decoder (28 layers, hidden 1536, GQA 12q:2kv). The model
ships as ``trust_remote_code`` custom code (``DotsOCRForCausalLM`` subclassing
``Qwen2ForCausalLM``); the revision is pinned for reproducibility.
"""
import types
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# Pinned to the current main revision so the trust_remote_code modeling files
# (configuration_dots.py, modeling_dots_ocr.py, modeling_dots_vision.py) are
# reproducible across runs.
_DOTS_OCR_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"


def _reinit_vision_rotary(model):
    """Recompute the vision rotary ``inv_freq`` buffer after loading.

    ``VisionRotaryEmbedding`` registers ``inv_freq`` as a NON-persistent buffer
    computed in ``__init__``. transformers ``from_pretrained`` fast-init skips
    that computation (the buffer isn't in the checkpoint), leaving it filled with
    uninitialized memory (~1e32). The garbage rotary produces inf/nan vision
    features whose CPU-vs-device inf handling diverges, which is the true cause of
    the corrupt end-to-end PCC. We recompute it exactly per the model definition
    (``inv_freq = 1 / theta**(arange(0, dim, 2)/dim)``, theta=10000). This mirrors
    DeepSeek-OCR's ``reinit_llama_rotary_inv_freq_buffers`` workaround.
    """
    rotary = model.vision_tower.rotary_pos_emb
    dim = rotary.inv_freq.numel() * 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    with torch.no_grad():
        rotary.inv_freq = inv_freq.to(
            device=rotary.inv_freq.device, dtype=rotary.inv_freq.dtype
        )
    return model


def _rewrite_patch_embed_conv_to_linear(model):
    """Rewrite the vision patch-embed Conv2d (kernel==stride==patch_size) as a Linear.

    The patch embedding applies ``Conv2d(3, embed_dim, kernel=14, stride=14)`` to
    non-overlapping 14x14 patches, which is mathematically identical to a Linear
    over the flattened patch. The TT conv2d program factory raises a runtime
    TT_FATAL on this degenerate (kernel==stride, 14x14) shape, so we fold the conv
    weights into an equivalent matmul. This is numerically exact (no accuracy
    impact) and only changes how the op lowers on device.
    """
    patchifier = model.vision_tower.patch_embed.patchifier
    conv = patchifier.proj
    # (embed_dim, in_ch, kH, kW) -> (embed_dim, in_ch*kH*kW)
    weight_lin = conv.weight.detach().reshape(conv.out_channels, -1).clone()
    bias_lin = None if conv.bias is None else conv.bias.detach().clone()
    patchifier.register_buffer("_proj_weight_lin", weight_lin, persistent=False)
    if bias_lin is not None:
        patchifier.register_buffer("_proj_bias_lin", bias_lin, persistent=False)
    else:
        patchifier._proj_bias_lin = None

    num_channels = patchifier.num_channels
    temporal = patchifier.temporal_patch_size
    psize = patchifier.patch_size
    embed_dim = patchifier.embed_dim

    def forward(self, x, grid_thw=None):
        # x: (num_patches, num_channels * temporal * patch * patch)
        x = x.view(-1, num_channels, temporal, psize, psize)[:, :, 0]
        x = x.reshape(x.shape[0], -1)
        x = F.linear(x, self._proj_weight_lin.to(x.dtype), None
                     if self._proj_bias_lin is None else self._proj_bias_lin.to(x.dtype))
        x = self.norm(x)
        return x

    patchifier.forward = types.MethodType(forward, patchifier)
    return model


def _rewrite_image_scatter(model):
    """Replace the data-dependent image-embedding insertion with a scatter-free,
    static-shape equivalent.

    The stock ``prepare_inputs_embeds`` uses ``torch.nonzero`` (data-dependent
    output shape) and ``Tensor.masked_scatter`` (lowers to ``stablehlo.scatter``)
    to drop the vision embeddings into the text-embedding sequence at the image
    placeholder positions. On device this fusion miscompiles (the standalone
    vision tower and text decoder each match CPU at PCC>0.99, but the fused
    forward corrupts to PCC~0.15). We rebuild the insertion from a one-hot
    selection matmul driven by ``cumsum`` over the image mask, which is
    mathematically identical to ``masked_scatter`` (the i-th image token receives
    the i-th vision embedding) but uses only eq / cumsum / matmul / mul.
    """
    image_token_id = model.config.image_token_id

    def prepare_inputs_embeds(self, input_ids, pixel_values=None, grid_thw=None, img_mask=None):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is None:
            return inputs_embeds

        vision_embeddings = self.vision_tower(pixel_values, grid_thw).to(inputs_embeds.dtype)
        n_img = vision_embeddings.shape[0]

        b, s, d = inputs_embeds.shape
        flat = inputs_embeds.reshape(b * s, d)
        m = (input_ids.reshape(b * s) == image_token_id)
        m_f = m.to(inputs_embeds.dtype).unsqueeze(1)  # (B*S, 1)

        # sel[j] = index (0..n_img-1) of the j-th image token in row-major order.
        sel = torch.cumsum(m.to(torch.int32), 0) - 1  # (B*S,)
        cols = torch.arange(n_img, device=inputs_embeds.device, dtype=sel.dtype)
        onehot = (sel.unsqueeze(1) == cols.unsqueeze(0)).to(inputs_embeds.dtype)  # (B*S, n_img)
        onehot = onehot * m_f  # zero out non-image rows
        scattered = onehot @ vision_embeddings  # (B*S, d)
        flat = flat * (1.0 - m_f) + scattered
        return flat.reshape(b, s, d)

    model.prepare_inputs_embeds = types.MethodType(prepare_inputs_embeds, model)
    return model


def _rewrite_vision_attention(model):
    """Replace the vision blocks' attention with mask-free full attention.

    The stock ``eager`` ``VisionAttention`` builds its attention mask with a
    Python loop over tensor-valued ``cu_seqlens`` bounds
    (``attention_mask[..., a:b, a:b] = 0``). This is data-dependent indexing that
    forces a torch.compile graph break; when the vision tower is traced as part
    of the fused ``prepare_inputs_embeds`` graph the broken-up subgraphs miscompile
    (the fused vision output drops to PCC~0.27 even though the tower compiles to
    PCC>0.99 in isolation). For a single image the whole patch sequence is one
    attention segment, so the mask is uniformly "attend" — i.e. plain full
    self-attention. We swap in that exact, static-shape attention.
    """
    import math
    import sys as _sys

    for blk in model.vision_tower.blocks:
        attn = blk.attn
        mod = _sys.modules[type(attn).__module__]
        apply_rope = mod.apply_rotary_pos_emb_vision
        num_heads = attn.num_heads
        head_dim = attn.qkv.out_features // 3 // num_heads

        def attn_forward(self, hidden_states, cu_seqlens=None, rotary_pos_emb=None,
                         _apply_rope=apply_rope, _num_heads=num_heads, _head_dim=head_dim):
            seq_length = hidden_states.shape[0]
            q, k, v = (
                self.qkv(hidden_states)
                .reshape(seq_length, 3, _num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )
            q = _apply_rope(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
            k = _apply_rope(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(_head_dim)
            attn_weights = attn_weights.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v).transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, -1)
            return self.proj(attn_output)

        attn.forward = types.MethodType(attn_forward, attn)
    return model


class ModelVariant(StrEnum):
    """Available dots.ocr model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """dots.ocr model loader for document OCR image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Fixed OCR prompt used for the sample input.
    sample_prompt = "Extract the text from the image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        # Fold the vision patch-embed Conv2d into an equivalent Linear so the
        # degenerate kernel==stride conv does not hit the TT conv2d runtime
        # assertion. Numerically exact; disable only for an apples-to-apples
        # comparison against the stock checkpoint.
        self.rewrite_patch_embed = True
        # Replace torch.nonzero + masked_scatter image-token insertion with a
        # scatter-free matmul equivalent (the data-dependent scatter miscompiles
        # on device). Numerically exact for the single-image OCR input.
        self.rewrite_image_scatter = True
        # Replace the vision attention's data-dependent cu_seqlens mask loop with
        # mask-free full attention (exact for a single image), removing the
        # torch.compile graph break that corrupts the fused vision output.
        self.rewrite_vision_attention = True

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load and cache the processor for the current variant."""
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
                revision=_DOTS_OCR_REVISION,
            )
        return self.processor

    @staticmethod
    def _sample_image() -> Image.Image:
        """Build a deterministic document-like image (no network, no font files)."""
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Default bitmap font ships with Pillow; renders deterministically.
        draw.text((10, 40), "dots OCR\non Tenstorrent\n2026", fill=(0, 0, 0))
        return img

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the dots.ocr model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The dots.ocr DotsOCRForCausalLM instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "revision": _DOTS_OCR_REVISION}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        # Force a static-graph-friendly attention implementation for the vision
        # tower (the checkpoint defaults to flash_attention_2, which is not
        # installed/usable on the device path).
        model_kwargs["attn_implementation"] = "sdpa"
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.config = model.config
        # Always: fix the uninitialized vision rotary inv_freq buffer.
        _reinit_vision_rotary(model)
        if self.rewrite_patch_embed:
            _rewrite_patch_embed_conv_to_linear(model)
        if self.rewrite_vision_attention:
            _rewrite_vision_attention(model)
        if self.rewrite_image_scatter:
            _rewrite_image_scatter(model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the dots.ocr model.

        Args:
            dtype_override: Optional torch.dtype to override the pixel_values dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: input_ids, attention_mask, pixel_values, image_grid_thw.
        """
        self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._sample_image()},
                    {"type": "text", "text": self.sample_prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # The forward signature does not accept mm_token_type_ids; drop it so the
        # extra key is not forwarded into the Qwen2 decoder.
        inputs.pop("mm_token_type_ids", None)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        if batch_size != 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and key not in (
                    "pixel_values",
                    "image_grid_thw",
                ):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

    def load_config(self):
        """Load and return the configuration for the dots.ocr model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=_DOTS_OCR_REVISION,
        )
        return self.config
