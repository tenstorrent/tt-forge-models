# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Janus-Pro multimodal-understanding (image+text -> text) loader helpers.

The understanding path differs from text-to-image generation: an image is
encoded by the SigLIP vision tower and projected by the (understanding)
``aligner`` into language space, spliced into the text-token embeddings, and
the LLaMA ``language_model`` then produces text logits through its ``lm_head``
(reference: deepseek-ai/Janus ``inference.py``).

Weight loading, the deepseek-ai/Janus runtime package and all transformers
compatibility patches are shared with the text-to-image component loader; this
module reuses them and only adds understanding-specific forge inputs.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

# Reuse the text-to-image loader's weight-loading + transformers-compat helpers.
from ....text_to_image.pytorch.src.model_utils import (  # noqa: E501
    DEVICE,
    DTYPE,
    load_mmgpt,
    load_processor,
)

REPO_ID_PRO_1B = "deepseek-ai/Janus-Pro-1B"
REPO_ID_PRO_7B = "deepseek-ai/Janus-Pro-7B"

IMG_SIZE = 384
# Number of <image_placeholder> tokens the processor expands per image.
IMAGE_TOKEN_NUM_PER_IMAGE = 576

UNDERSTAND_PROMPT = (
    "<image_placeholder>\nDescribe this image in detail."
)


def _sample_image(size: int = IMG_SIZE) -> Image.Image:
    """Deterministic synthetic RGB image (no network/asset dependency).

    The understanding forward pass is shape/dtype driven; a fixed-seed image
    keeps inputs reproducible across CPU and device runs.
    """
    rng = np.random.RandomState(42)
    arr = (rng.rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


def build_understanding_prepare(repo_id: str, prompt: str = UNDERSTAND_PROMPT):
    """Run VLChatProcessor on a (image, prompt) turn -> batchified inputs."""
    processor = load_processor(repo_id)
    image = _sample_image()
    conversation = [
        {"role": "<|User|>", "content": prompt, "images": [image]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    return processor(
        conversations=conversation,
        images=[image],
        force_batchify=True,
    )


@torch.inference_mode()
def make_understanding_inputs_embeds(
    repo_id: str,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    """Build multimodal inputs_embeds on CPU (vision tower + aligner + text embeds).

    Returns the combined ``[1, T, D]`` sequence ready for the language model.
    """
    dtype = dtype if dtype is not None else DTYPE
    mmgpt = load_mmgpt(repo_id, dtype)
    prepare = build_understanding_prepare(repo_id)
    inputs_embeds = mmgpt.prepare_inputs_embeds(
        input_ids=prepare.input_ids.to(DEVICE),
        pixel_values=prepare.pixel_values.to(device=DEVICE, dtype=dtype),
        images_seq_mask=prepare.images_seq_mask.to(DEVICE),
        images_emb_mask=prepare.images_emb_mask.to(DEVICE),
    )
    return {"inputs_embeds": inputs_embeds.to(dtype=dtype)}


def make_vision_embed_inputs(
    repo_id: str,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    """Pixel inputs for the vision-tower + aligner component: ``[bn, 3, H, W]``."""
    dtype = dtype if dtype is not None else DTYPE
    prepare = build_understanding_prepare(repo_id)
    # pixel_values arrive as [b, n_images, 3, H, W]; flatten the image axis.
    pixel_values = prepare.pixel_values.reshape(-1, 3, IMG_SIZE, IMG_SIZE)
    return {"pixel_values": pixel_values.to(device=DEVICE, dtype=dtype)}
