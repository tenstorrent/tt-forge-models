#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XLA compatibility patches for the Wan 2.2 VAE.

The diffusers AutoencoderKLWan uses ``x[:, :, -CACHE_T:, :, :]`` (CACHE_T=2)
to extract temporal cache slices.  When processing single-frame chunks, the
temporal dimension is 1 and the negative index -2 is out of range for XLA
(PyTorch clamps it to 0 silently, XLA does not).

This module patches the installed diffusers source to replace the problematic
slice with ``x[:, :, max(0, x.shape[2]-CACHE_T):, :, :]`` which is equivalent
but XLA-safe.
"""

import re
from pathlib import Path

_PATTERN = r"x\[:, :, -CACHE_T:, :, :\]"
_REPLACEMENT = "x[:, :, max(0, x.shape[2]-CACHE_T):, :, :]"


def patch_wan_vae_for_xla():
    """Patch the installed diffusers AutoencoderKLWan source for XLA compatibility."""
    import diffusers.models.autoencoders.autoencoder_kl_wan as mod

    src_path = Path(mod.__file__)
    src = src_path.read_text()

    if _REPLACEMENT in src:
        return

    patched = re.sub(_PATTERN, _REPLACEMENT, src)
    if patched == src:
        return

    src_path.write_text(patched)

    # Remove cached bytecode so Python picks up the patched source
    pyc = src_path.with_suffix(".pyc")
    if pyc.exists():
        pyc.unlink()
    cache_dir = src_path.parent / "__pycache__"
    if cache_dir.exists():
        for f in cache_dir.glob(f"{src_path.stem}*.pyc"):
            f.unlink()
