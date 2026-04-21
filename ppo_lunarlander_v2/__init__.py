# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PPO LunarLander-v2 reinforcement learning model."""
# Import from the PyTorch implementation by default
from .pytorch import ModelLoader, ModelVariant

__all__ = ["ModelLoader", "ModelVariant"]
