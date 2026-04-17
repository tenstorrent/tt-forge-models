# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Compatibility shim: re-export ``Module`` so that HuggingFace remote model code
using ``from sentence_transformers.models import Module`` works with
sentence-transformers >= 5 (which moved ``Module`` to
``sentence_transformers.base.modules.module``).
"""
import torch
from abc import ABC


class Module(ABC, torch.nn.Module):
    pass
