# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helpers for Janus-Pro T2I bring-up: HF weight load, processor, CFG prompt embeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import JanusForConditionalGeneration, JanusProcessor


STANDARD_PROMPT = (
    "A close-up high-contrast photo of Sydney Opera House sitting next to "
    "Eiffel tower, under a blue night sky of roiling energy, exploding yellow "
    "stars, and radiating swirls of blue."
)


def load_processor(repo_id: str) -> JanusProcessor:
    """Load JanusProcessor for a Hub repo id."""
    from transformers import JanusProcessor

    return JanusProcessor.from_pretrained(repo_id)


def load_janus_model(
    repo_id: str,
    *,
    dtype_override: torch.dtype | None = None,
    **kwargs,
) -> JanusForConditionalGeneration:
    """Load Janus-Pro weights (backbone for input prep; compile target is src.model wrapper)."""
    from transformers import JanusForConditionalGeneration

    model_kwargs: dict = {
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if dtype_override is not None:
        model_kwargs["torch_dtype"] = dtype_override
    model_kwargs.update(kwargs)

    model = JanusForConditionalGeneration.from_pretrained(repo_id, **model_kwargs)
    model.eval()
    if dtype_override is not None:
        model = model.to(dtype=dtype_override)
    return model


def prepare_cfg_prompt_embeds(
    model: JanusForConditionalGeneration,
    processor: JanusProcessor,
    dtype_override: torch.dtype | None,
    *,
    prompt: str = STANDARD_PROMPT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CFG-doubled prompt embeds for image generate() step 0 (matches HF Janus generate)."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(text=prompt_text, generation_mode="image", return_tensors="pt")
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask

    gc = model.generation_config
    batch_size = 1
    input_tokens = input_ids.repeat(2, 1)
    attention_mask = attention_mask.repeat(2, 1)
    boi_id = gc.generation_kwargs["boi_token_id"]
    mask = (input_tokens[batch_size:, :] != gc.bos_token_id) & (
        input_tokens[batch_size:, :] != boi_id
    )
    input_tokens[batch_size:, :].masked_fill_(mask, gc.pad_token_id)
    inputs_embeds = model.get_input_embeddings()(input_tokens)
    if dtype_override is not None:
        inputs_embeds = inputs_embeds.to(dtype=dtype_override)
    return inputs_embeds, attention_mask


def load_cfg_prompt_inputs(
    model: JanusForConditionalGeneration,
    processor: JanusProcessor,
    dtype_override: torch.dtype | None,
    *,
    prompt: str = STANDARD_PROMPT,
) -> dict[str, torch.Tensor]:
    """Inputs for JanusImageTokenLogitsStep (CFG batch size 2)."""
    inputs_embeds, attention_mask = prepare_cfg_prompt_embeds(
        model, processor, dtype_override, prompt=prompt
    )
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }
