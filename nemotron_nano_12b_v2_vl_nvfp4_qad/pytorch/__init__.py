# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Nemotron Nano 12B v2 VL NVFP4-QAD PyTorch model loader implementation.
"""
from .loader import ModelLoader, ModelVariant

__all__ = ["ModelLoader", "ModelVariant"]
