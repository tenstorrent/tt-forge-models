# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Cambrian model utilities: registers the cambrian_qwen architecture with
transformers and patches compatibility issues with newer transformers versions.
"""

import torch.nn as nn
from transformers import Qwen2ForCausalLM
from cambrian.model import CambrianQwenForCausalLM, CambrianQwenConfig  # noqa: F401
import cambrian.model.language_model.cambrian_qwen2 as _cq2


def _patched_init(self, config):
    """Patched __init__ that prevents config.rope_scaling = None from
    clearing rope_parameters in transformers >= 5."""
    Qwen2ForCausalLM.__init__(self, config)
    config.model_type = "cambrian_qwen"
    # Save rope_parameters before rope_scaling setter clears it
    saved_rope_params = getattr(config, "rope_parameters", None)
    config.rope_scaling = None
    # Restore rope_parameters
    if getattr(config, "rope_parameters", None) is None:
        if saved_rope_params is not None:
            config.rope_parameters = saved_rope_params
        else:
            config.rope_parameters = {"rope_type": "default"}
            if hasattr(config, "rope_theta"):
                config.rope_parameters["rope_theta"] = config.rope_theta

    self.model = _cq2.CambrianQwenModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()


_cq2.CambrianQwenForCausalLM.__init__ = _patched_init
