# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Janus-Pro helpers: weights load, CFG prompt embeds, transformers 5.x load patches.

5.x patches are scoped: torch.linspace is restored after each load; post_init patch
applies only to janus.models.MultiModalityCausalLM (no effect on other forge models).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM

REPO_ID_PRO_1B = "deepseek-ai/Janus-Pro-1B"
REPO_ID_PRO_7B = "deepseek-ai/Janus-Pro-7B"

DTYPE = torch.bfloat16
DEVICE = "cpu"
PARALLEL_SIZE = 1

STANDARD_PROMPT = (
    "A close-up high-contrast photo of Sydney Opera House sitting next to "
    "Eiffel tower, under a blue night sky of roiling energy, exploding yellow "
    "stars, and radiating swirls of blue."
)

_orig_torch_linspace = None
_processor_cache: dict[str, VLChatProcessor] = {}
_mmgpt_cache: dict[str, MultiModalityCausalLM] = {}


def transformers_major_version() -> int:
    import transformers

    return int(transformers.__version__.split(".")[0])


def needs_5x_load_patches() -> bool:
    return transformers_major_version() >= 5


@contextmanager
def linspace_meta_patch_context() -> Iterator[None]:
    """
    Temporarily patch torch.linspace for janus siglip __init__ on transformers 5.x.

    Always restores torch.linspace on exit so other models in the same CI process
    are unaffected.
    """
    global _orig_torch_linspace
    if not needs_5x_load_patches():
        yield
        return
    if _orig_torch_linspace is not None:
        # Nested load (e.g. 1B then 7B): outer context owns restore
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
    """Idempotent; only touches janus MultiModalityCausalLM.__init__."""
    if not needs_5x_load_patches():
        return
    from janus.models import modeling_vlm

    cls = modeling_vlm.MultiModalityCausalLM
    if getattr(cls, "_janus_52_post_init_patched", False):
        return
    _orig_init = cls.__init__

    def __init__(self, config, *args, **kwargs):
        _orig_init(self, config, *args, **kwargs)
        self.post_init()

    cls.__init__ = __init__
    cls._janus_52_post_init_patched = True


def model_from_pretrained_kwargs() -> dict:
    if needs_5x_load_patches():
        return {"low_cpu_mem_usage": False}
    return {"low_cpu_mem_usage": True}


def load_processor(repo_id: str) -> VLChatProcessor:
    if repo_id not in _processor_cache:
        _processor_cache[repo_id] = VLChatProcessor.from_pretrained(repo_id)
    return _processor_cache[repo_id]


def load_mmgpt(
    repo_id: str,
    dtype: torch.dtype = DTYPE,
    **kwargs,
) -> MultiModalityCausalLM:
    if repo_id not in _mmgpt_cache:
        apply_multimodality_post_init_patch()
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


def build_prompt(vl_chat_processor: VLChatProcessor, user_prompt: str) -> str:
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
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
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
