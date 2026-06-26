# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2-8B component utilities.

Molmo2 (``Molmo2ForConditionalGeneration``) is a vision-language model that pairs
a SigLIP-style vision tower + pooling/projector adapter with a Qwen3-8B-style
text decoder. It ships as custom (``trust_remote_code``) code on the Hub.

For bringup we validate the two compute components independently, the same way a
diffusion pipeline is brought up per-component:

  * VISION_TOWER  -> ``Molmo2VisionTransformer`` (the image ViT), a single
                     forward pass over flattened image patches.
  * TEXT_DECODER  -> ``Molmo2TextModel`` + ``lm_head``, a single causal-LM
                     forward pass over token ids producing vocab logits.

Both components are sliced out of one ``from_pretrained`` load of the full model
so they carry the real pretrained weights.
"""

from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# Pin the model + remote-code revision so a future run provisions the exact same
# architecture/weights instead of silently picking up a new custom-code drop.
REPO_ID = "allenai/Molmo2-8B"
REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

# Original weights are distributed as fp32 on the Hub; we run in bf16 on device.
DTYPE = torch.bfloat16

# Text-decoder sample-input sequence-length cap (kept short for the device path).
# Inputs are NOT padded: a fully-masked padding row makes softmax produce NaN in
# the CPU reference, which poisons the PCC comparison. We use the prompt's natural
# token length (truncated to this cap), so every attention row has a real token.
TEXT_SEQ_LEN = 32

# A fixed prompt for deterministic text-decoder PoC decoding. Chosen long enough
# to exercise a non-trivial sequence length without padding.
TEXT_PROMPT = (
    "Tenstorrent builds AI accelerators and the open-source software stack that "
    "runs large language models on them. The capital of France is"
)


def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    """Standard (un-scaled) RoPE inverse frequencies.

    transformers >= 5.x dropped the ``"default"`` entry from
    ``ROPE_INIT_FUNCTIONS`` during the RoPE refactor, but Molmo2's pinned custom
    code still looks it up (``rope_type == "default"``). Re-provide the classic
    computation so the model loads on the installed transformers version.
    """
    config.standardize_rope_params()
    rope_parameters_dict = (
        config.rope_parameters[layer_type]
        if layer_type is not None
        else config.rope_parameters
    )
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0


def _patch_rope_default():
    """Register the ``"default"`` RoPE init fn if the transformers version omits it."""
    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _fix_rotary_buffers(model):
    """Recompute non-persistent RoPE ``inv_freq`` buffers after loading.

    transformers >= 5.x's weight loader does not materialize buffers that are
    absent from the checkpoint (``inv_freq`` is registered ``persistent=False``),
    leaving them as uninitialized garbage -> NaN cos/sin -> NaN logits. The
    buffer is correct after ``__init__`` but corrupted by ``from_pretrained``, so
    we recompute it from each rotary module's own ``rope_init_fn``.
    """
    for module in model.modules():
        if hasattr(module, "rope_init_fn") and hasattr(module, "inv_freq"):
            inv_freq, attention_scaling = module.rope_init_fn(
                module.config, module.inv_freq.device
            )
            inv_freq = inv_freq.to(module.inv_freq.dtype)
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.original_inv_freq = inv_freq
            if hasattr(module, "attention_scaling"):
                module.attention_scaling = attention_scaling


def load_full_model(dtype: torch.dtype):
    """Load the full Molmo2ForConditionalGeneration model in eval mode."""
    _patch_rope_default()
    model = AutoModelForImageTextToText.from_pretrained(
        REPO_ID,
        revision=REVISION,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    _fix_rotary_buffers(model)
    return model.eval()


class TextDecoderWrapper(torch.nn.Module):
    """Molmo2 text decoder (``Molmo2TextModel``) + ``lm_head`` -> vocab logits.

    A single causal-LM forward pass: token ids in, next-token logits out. KV
    cache is disabled so the graph is a static-shape prefill, which is what the
    device path compiles.
    """

    def __init__(self, text_model: torch.nn.Module, lm_head: torch.nn.Module):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self.lm_head(hidden)


class VisionTowerWrapper(torch.nn.Module):
    """Molmo2 image ViT (``Molmo2VisionTransformer``) -> final hidden state.

    The ViT forward returns a per-layer list of hidden states; we expose the
    final layer's output (the deepest feature map) as a single tensor for the
    PCC comparison.
    """

    def __init__(self, image_vit: torch.nn.Module):
        super().__init__()
        self.image_vit = image_vit

    def forward(self, pixel_values: torch.Tensor):
        hidden_states = self.image_vit(pixel_values)
        return hidden_states[-1]


def load_text_decoder(dtype: torch.dtype) -> TextDecoderWrapper:
    """Slice the text decoder + lm_head out of the full model."""
    model = load_full_model(dtype)
    wrapper = TextDecoderWrapper(model.model.transformer, model.lm_head)
    return wrapper.eval()


def load_vision_tower(dtype: torch.dtype) -> VisionTowerWrapper:
    """Slice the image ViT out of the full model's vision backbone."""
    model = load_full_model(dtype)
    image_vit = model.model.vision_backbone.image_vit
    wrapper = VisionTowerWrapper(image_vit)
    return wrapper.eval()


def load_text_decoder_inputs(dtype: torch.dtype) -> dict:
    """Tokenize a fixed prompt, padded/truncated to ``TEXT_SEQ_LEN``.

    Returns ``input_ids`` / ``attention_mask`` of shape ``[1, TEXT_SEQ_LEN]``
    (int64). The dtype override does not apply to integer ids.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ID, revision=REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        TEXT_PROMPT,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=TEXT_SEQ_LEN,
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }


def load_vision_tower_inputs(dtype: torch.dtype) -> dict:
    """Synthetic flattened image patches for the ViT.

    Shape ``[1, image_num_pos, patch_size*patch_size*3]`` = ``[1, 729, 588]`` for
    the 378x378 / patch-14 configuration. A deterministic random tensor is used
    (the ViT here is a feature extractor with no classifier head, so the PoC is a
    feature-map PCC rather than a label).
    """
    torch.manual_seed(0)
    # image_num_pos = 729 (27x27 patches), n_pixels = 14*14*3 = 588.
    pixel_values = torch.randn(1, 729, 588, dtype=dtype)
    return {"pixel_values": pixel_values}
