# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for the rednote-hilab/dots.ocr loaders.

dots.ocr is a document-OCR vision-language model: a NaViT-style vision tower
(``DotsVisionTransformer``, model_type ``dots_vit``) feeding a Qwen2 decoder
(``DotsOCRForCausalLM`` subclasses ``Qwen2ForCausalLM``). The model ships as
custom HuggingFace code (``trust_remote_code=True``); we pin a revision so the
remote modeling files are reproducible.

The three loaders under this family share weight/processor loading here:
  * ``mm_doc_ocr`` - full end-to-end OCR forward (vision tower + decoder glue).
  * ``vision_tower`` - the ``DotsVisionTransformer`` image encoder alone.
  * ``causal_lm`` - the Qwen2 text decoder alone (text-only inputs).
"""
import torch
from PIL import Image, ImageDraw

# Pinned commit of rednote-hilab/dots.ocr (custom remote code + weights).
DOTS_OCR_MODEL = "rednote-hilab/dots.ocr"
DOTS_OCR_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"

# Default OCR prompt shipped in the dots.ocr model card (layout/markdown parse).
SAMPLE_PROMPT = "Extract the text content from this image."

# Vision input is bounded so the vision-tower attention (O(seq^2) over patches)
# stays compilable on a single chip. With patch_size=14 and spatial_merge=2 the
# pixel grid must be a multiple of 28; 308x308 -> a 22x22 patch grid (484 tokens).
DEFAULT_IMAGE_SIZE = 308


def make_document_image(size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    """Deterministic synthetic 'document' image (no network dependency).

    Content is irrelevant for op-support / PCC validation; a fixed white page
    with a few lines of black text gives the processor a realistic layout to
    patchify.
    """
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "Tenstorrent dots.ocr bringup",
        "Document OCR validation page.",
        "Line three: 0123456789.",
        "The quick brown fox jumps.",
    ]
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 40
    return img


def load_processor():
    """Load the DotsVLProcessor (Qwen2_5_VLProcessor subclass) for dots.ocr."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        DOTS_OCR_MODEL,
        revision=DOTS_OCR_REVISION,
        trust_remote_code=True,
    )


def load_full_model(dtype_override=None):
    """Load the full DotsOCRForCausalLM (vision tower + Qwen2 decoder).

    The vision tower defaults to ``flash_attention_2``; since flash-attn is not
    installed it self-falls-back to the eager (plain-matmul) attention, which is
    the static-shape path we want on device.
    """
    from transformers import AutoModelForCausalLM

    model_kwargs = {"trust_remote_code": True, "revision": DOTS_OCR_REVISION}
    if dtype_override is not None:
        model_kwargs["torch_dtype"] = dtype_override

    model = AutoModelForCausalLM.from_pretrained(DOTS_OCR_MODEL, **model_kwargs)
    model.config.use_cache = False
    _reinit_vision_rotary(model)
    model.eval()
    return model


def _reinit_vision_rotary(model, theta: float = 10000.0):
    """Recompute the vision tower's ``VisionRotaryEmbedding.inv_freq``.

    ``inv_freq`` is registered ``persistent=False`` and computed in __init__, so
    it is absent from the checkpoint. Under transformers' low-cpu-mem (meta) load
    path it comes back uninitialized (denormal garbage), which silently corrupts
    the 2D vision rotary. Recompute it deterministically so the vision tower
    produces correct embeddings.
    """
    rope = getattr(model.vision_tower, "rotary_pos_emb", None)
    if rope is None or not hasattr(rope, "inv_freq"):
        return
    n = rope.inv_freq.numel()
    dim = n * 2  # VisionRotaryEmbedding was built with dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    with torch.no_grad():
        rope.inv_freq = inv_freq.to(
            device=rope.inv_freq.device, dtype=torch.float32
        )


def compute_vision_rotary(grid_thw, head_dim=128, spatial_merge_size=2, theta=10000.0):
    """Host-side replica of DotsVisionTransformer.rot_pos_emb.

    The vision tower derives 2D rotary position embeddings purely from the
    (static) patch grid via ``torch.arange``/``.max()`` over per-image h/w. Those
    ops can't run against a device-resident ``grid_thw`` (they materialize device
    scalars), so we precompute the rotary table on the host and feed it into the
    device graph as a plain tensor. Mirrors ``get_pos_ids_by_grid`` +
    ``VisionRotaryEmbedding`` exactly so device output matches the reference.

    Returns a [num_patches, head_dim // 2] float32 tensor.
    """
    grid_thw = torch.as_tensor(grid_thw, dtype=torch.long, device="cpu")
    m = spatial_merge_size
    pos_ids = []
    for t, h, w in grid_thw.tolist():
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)

    max_grid_size = int(grid_thw[:, 1:].max())
    dim = head_dim // 2  # VisionRotaryEmbedding dim
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    seq = torch.arange(max_grid_size, dtype=inv_freq.dtype)
    rotary_full = torch.outer(seq, inv_freq)  # [max_grid, dim//2]
    rotary_pos_emb = rotary_full[pos_ids].flatten(1)  # [num_patches, dim]
    return rotary_pos_emb


def build_multimodal_inputs(processor, prompt=SAMPLE_PROMPT, image_size=DEFAULT_IMAGE_SIZE,
                            dtype_override=None):
    """Build full image+text inputs (input_ids, attention_mask, pixel_values,
    image_grid_thw) via the chat template, the way DotsVLProcessor expects."""
    image = make_document_image(image_size)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = dict(inputs)
    if dtype_override is not None and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)
    return inputs
