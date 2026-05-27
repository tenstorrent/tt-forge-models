# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Janus-Pro loader helpers: weight load, compatibility patches, forge test inputs.

Load-time patches are always applied when loading ``MultiModalityCausalLM`` (they are
not gated on a transformers major version). They address Janus + current Hugging Face
loading behavior in tt-xla (today transformers 5.2.x); the same hooks are harmless on
older transformers releases where the issues did not appear.

Patches are scoped to Janus load only: ``torch.linspace`` is restored after each load;
``post_init`` is injected only on ``janus.models.MultiModalityCausalLM``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
from transformers import AutoModelForCausalLM

REPO_ID_PRO_1B = "deepseek-ai/Janus-Pro-1B"
REPO_ID_PRO_7B = "deepseek-ai/Janus-Pro-7B"

DTYPE = torch.bfloat16
DEVICE = "cpu"
PARALLEL_SIZE = 1
IMAGE_TOKEN_NUM_PER_IMAGE = 576
IMG_SIZE = 384
PATCH_SIZE = 16
CODEBOOK_DIM = 8

STANDARD_PROMPT = (
    "A close-up high-contrast photo of Sydney Opera House sitting next to "
    "Eiffel tower, under a blue night sky of roiling energy, exploding yellow "
    "stars, and radiating swirls of blue."
)

_orig_torch_linspace = None
_processor_cache: dict[str, Any] = {}
_mmgpt_cache: dict[str, Any] = {}


def decode_shape(parallel_size: int = PARALLEL_SIZE) -> list[int]:
    latent = IMG_SIZE // PATCH_SIZE
    return [parallel_size, CODEBOOK_DIM, latent, latent]


def apply_janus_load_patches() -> None:
    """
    Apply Janus-specific load compatibility hooks before ``from_pretrained``.

    Always enabled for this loader (not tied to a transformers version check). Issues
    were first observed when tt-xla moved to transformers 5.x; the patches remain
    valid for whatever transformers version the forge env provides.
    """
    apply_multimodality_post_init_patch()


@contextmanager
def linspace_meta_patch_context() -> Iterator[None]:
    """
    Temporarily patch ``torch.linspace`` during Janus SigLIP init.

    When HF lazy loading leaves meta-device tensors, re-materialize on CPU. The wrapper
    is a no-op when ``linspace`` does not return meta tensors. Always restores the
    original ``torch.linspace`` on exit so other forge models are unaffected.
    """
    global _orig_torch_linspace
    if _orig_torch_linspace is not None:
        yield
        return

    _orig_torch_linspace = torch.linspace

    def _linspace(*args, **kwargs):
        out = _orig_torch_linspace(*args, **kwargs)
        if out.is_meta:
            return _orig_torch_linspace(*args, **kwargs, device="cpu")
        return out

    torch.linspace = _linspace
    try:
        yield
    finally:
        torch.linspace = _orig_torch_linspace
        _orig_torch_linspace = None


def apply_multimodality_post_init_patch() -> None:
    """Idempotent; only touches ``janus.models.MultiModalityCausalLM.__init__``."""
    from janus.models import modeling_vlm

    cls = modeling_vlm.MultiModalityCausalLM
    if getattr(cls, "_janus_post_init_patched", False):
        return
    _orig_init = cls.__init__

    def __init__(self, config, *args, **kwargs):
        _orig_init(self, config, *args, **kwargs)
        self.post_init()

    cls.__init__ = __init__
    cls._janus_post_init_patched = True


def model_from_pretrained_kwargs() -> dict:
    """Kwargs for reliable Janus weight materialization with current transformers."""
    return {"low_cpu_mem_usage": False}


def load_processor(repo_id: str):
    from janus.models import VLChatProcessor

    if repo_id not in _processor_cache:
        _processor_cache[repo_id] = VLChatProcessor.from_pretrained(repo_id)
    return _processor_cache[repo_id]


def load_mmgpt(repo_id: str, dtype: torch.dtype = DTYPE, **kwargs):
    if repo_id not in _mmgpt_cache:
        apply_janus_load_patches()
        load_kw = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "attn_implementation": "eager",
            **model_from_pretrained_kwargs(),
            **kwargs,
        }
        with linspace_meta_patch_context():
            _mmgpt_cache[repo_id] = AutoModelForCausalLM.from_pretrained(
                repo_id,
                **load_kw,
            )
        _mmgpt_cache[repo_id].eval()
    return _mmgpt_cache[repo_id]


def build_prompt(vl_chat_processor, user_prompt: str) -> str:
    conversation = [
        {"role": "<|User|>", "content": user_prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag


def prepare_cfg_inputs_embeds(
    mmgpt,
    vl_chat_processor,
    prompt: str,
    *,
    parallel_size: int = PARALLEL_SIZE,
    device: str | torch.device = DEVICE,
) -> torch.Tensor:
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(
        device
    )
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    return mmgpt.language_model.get_input_embeddings()(tokens)


def make_cfg_inputs_embeds(
    repo_id: str,
    dtype: Optional[torch.dtype] = None,
    *,
    prompt: str = STANDARD_PROMPT,
) -> torch.Tensor:
    dtype = dtype if dtype is not None else DTYPE
    processor = load_processor(repo_id)
    mmgpt = load_mmgpt(repo_id, dtype)
    prompt_full = build_prompt(processor, prompt)
    return prepare_cfg_inputs_embeds(
        mmgpt, processor, prompt_full, parallel_size=PARALLEL_SIZE, device=DEVICE
    ).to(dtype=dtype)


def _cfg_image_ids(next_token: torch.Tensor) -> torch.Tensor:
    """Duplicate sampled token for CFG cond/uncond rows (parallel_size=1)."""
    t = next_token.reshape(-1)
    return torch.cat([t, t], dim=0)


def align_kv_cache_device(past_key_values: Any, device: torch.device | str) -> Any:
    """
    Move transformers DynamicCache tensors to ``device``.

    Decode forge inputs build the cache via a CPU prefill (``make_image_token_decode_inputs``).
    TT/XLA compile then runs with activations on ``xla:0``; without aligning the cache,
    ``torch.cat`` in ``cache_utils.DynamicLayer.update`` hits mixed cpu/xla FakeTensor errors.
    """
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        return past_key_values
    for layer in past_key_values.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        layer.keys = layer.keys.to(device=device)
        layer.values = layer.values.to(device=device)
        layer.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
    return past_key_values


@torch.inference_mode()
def make_image_token_decode_inputs(
    repo_id: str,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, Any]:
    """
    Decode-step forge inputs: KV cache after one prefill forward + 1-token embeds.

    Uses argmax on conditional logits (deterministic; no sampling loop).
    """
    dtype = dtype if dtype is not None else DTYPE
    mmgpt = load_mmgpt(repo_id, dtype)
    inputs_embeds = make_cfg_inputs_embeds(repo_id, dtype)
    outputs = mmgpt.language_model.model(
        inputs_embeds=inputs_embeds,
        use_cache=True,
        past_key_values=None,
    )
    logits = mmgpt.gen_head(outputs.last_hidden_state[:, -1, :])
    next_token = logits[0::2].argmax(dim=-1)
    image_ids = _cfg_image_ids(next_token)
    step_embeds = mmgpt.prepare_gen_img_embeds(image_ids).unsqueeze(dim=1)
    return {
        "inputs_embeds": step_embeds.to(dtype=dtype),
        "past_key_values": outputs.past_key_values,
    }


def make_gen_img_embed_inputs(
    repo_id: str,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    del repo_id, dtype
    return {
        "image_ids": torch.zeros(PARALLEL_SIZE * 2, dtype=torch.long, device=DEVICE),
    }


def make_gen_vision_decode_inputs(
    repo_id: str,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    del repo_id, dtype
    return {
        "generated_tokens": torch.zeros(
            (PARALLEL_SIZE, IMAGE_TOKEN_NUM_PER_IMAGE),
            dtype=torch.int,
            device=DEVICE,
        ),
    }
