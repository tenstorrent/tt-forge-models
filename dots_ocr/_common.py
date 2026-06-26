# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for the rednote-hilab/dots.ocr loaders.

dots.ocr is a multimodal document-OCR model (``DotsOCRForCausalLM``) that ships
its modeling code via ``trust_remote_code``. It pairs a NaViT-style vision tower
(``dots_vit``, 42 layers) with a Qwen2 text decoder (28 layers). The three
loaders under this family bring the model up component-by-component:

* ``image_text_to_text`` - the full pipeline (vision tower + decoder, end-to-end)
* ``causal_lm``           - the Qwen2 text decoder in isolation (text-only forward)
* ``vision``              - the ``dots_vit`` vision tower in isolation

All loaders pin the same revision so a future run provisions the exact modeling
code we validated against.
"""
from PIL import Image, ImageDraw

# Pin the modeling code / weights revision so trust_remote_code is reproducible.
DOTS_OCR_MODEL = "rednote-hilab/dots.ocr"
DOTS_OCR_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"

# Image token used to splice vision embeddings into the text stream.
IMAGE_TOKEN_ID = 151665


def build_demo_image(size: int = 392) -> Image.Image:
    """Build a small, deterministic synthetic document image.

    ``size`` is chosen as a multiple of patch_size (14) * spatial_merge_size (2)
    = 28 so the Qwen2-VL image processor does not need to pad/resize, keeping the
    vision sequence length small and the grid exact (size/14 patches per side).
    A real network image is avoided so the bringup is hermetic and reproducible.
    """
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "Tenstorrent",
        "dots.ocr bringup",
        "Document OCR",
        "Invoice #2026-0626",
        "Total: $1,234.56",
    ]
    y = 24
    for line in lines:
        draw.text((24, y), line, fill=(0, 0, 0))
        y += 48
    return img
