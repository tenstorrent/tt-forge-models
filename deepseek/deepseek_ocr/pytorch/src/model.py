# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-OCR hub runtime: patch ``DeepSeek_OCR_hub/modeling_*.py`` + CPU ``.cuda()`` workaround.

Neural-net weights/code live in the hub snapshot (``loader.py``); this module does not fork ``modeling_*``.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
from pathlib import Path

import torch

# --- Hub source patches (applied to DeepSeek_OCR_hub/modeling_*.py) ---
_MASKED_SCATTER_OLD = (
    "inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), "
    "images_in_this_batch)"
)
_MASKED_SCATTER_NEW = """inputs_embeds[idx].masked_scatter(
                        images_seq_mask[idx].unsqueeze(-1),
                        images_in_this_batch,
                    )"""

_TORCH_SUM_IMAGES_OLD = "torch.sum(images[0][1]).item() != 0"
_TORCH_SUM_IMAGES_NEW = "torch.sum(images[0][1], dim=(0, 1, 2, 3)).item() != 0"

_MOE_NUMPY_OLD = "tokens_per_expert = tokens_per_expert.cpu().numpy()"
_MOE_NUMPY_NEW = "tokens_per_expert = tokens_per_expert.cpu().tolist()"

_CUDA_PATCHED = False


def _apply_text_patches(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if old not in text:
            continue
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_hub_snapshot(hub_dir: Path) -> bool:
    """Patch hub ``modeling_*.py`` for tt-xla. Returns True if any file changed."""
    any_changed = False
    ocr_py = hub_dir / "modeling_deepseekocr.py"
    v2_py = hub_dir / "modeling_deepseekv2.py"
    if ocr_py.is_file():
        any_changed |= _apply_text_patches(
            ocr_py,
            [
                (_MASKED_SCATTER_OLD, _MASKED_SCATTER_NEW),
                (_TORCH_SUM_IMAGES_OLD, _TORCH_SUM_IMAGES_NEW),
            ],
        )
    if v2_py.is_file():
        any_changed |= _apply_text_patches(v2_py, [(_MOE_NUMPY_OLD, _MOE_NUMPY_NEW)])
    return any_changed


def invalidate_transformers_hub_module_cache() -> None:
    """Drop HF dynamic-module cache so the next load uses patched files under ``hub_dir``."""
    cache_base = Path.home() / ".cache/huggingface/modules/transformers_modules"
    if cache_base.is_dir():
        for child in cache_base.iterdir():
            name = child.name.lower()
            if "deepseek" in name or "deepseek_ocr_hub" in name:
                shutil.rmtree(child, ignore_errors=True)

    to_drop = [
        k
        for k in list(sys.modules)
        if "deepseek" in k.lower()
        and ("DeepSeek" in k or "deepseek_hyphen" in k or "modeling_deepseek" in k)
    ]
    for k in to_drop:
        del sys.modules[k]


def prepare_hub_snapshot(hub_dir: Path) -> None:
    """Apply patches and invalidate cache (call before ``AutoModel.from_pretrained``)."""
    patch_hub_snapshot(hub_dir)
    invalidate_transformers_hub_module_cache()


def ensure_hub_cpu_cuda_workaround() -> None:
    """No-op ``Tensor.cuda`` on CPU-only PyTorch (hub code may call ``.cuda()`` elsewhere)."""
    global _CUDA_PATCHED
    if torch.cuda.is_available() or _CUDA_PATCHED:
        return
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[method-assign, assignment]
    _CUDA_PATCHED = True


@contextlib.contextmanager
def hub_cuda_workaround():
    """Context manager for scripts/tests (same as :func:`ensure_hub_cpu_cuda_workaround`)."""
    if torch.cuda.is_available():
        yield
        return
    ensure_hub_cpu_cuda_workaround()
    yield
