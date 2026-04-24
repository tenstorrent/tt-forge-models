# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Zonos TTS model loader implementation for text-to-speech tasks.
"""
import torch
import torch.nn as nn
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ZonosBackboneWrapper(nn.Module):
    """Wrapper around the Zonos backbone and output heads.

    Exposes a clean forward pass that takes pre-computed hidden states and
    produces per-codebook speech token logits.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states):
        import os
        import sys

        # Re-import InferenceParams from the installed zonos (not the local shadow).
        worktree_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        saved_path = sys.path[:]
        sys.path = [p for p in sys.path if p not in ("", ".", worktree_dir)]
        for k in [
            k for k in list(sys.modules) if k == "zonos" or k.startswith("zonos.")
        ]:
            del sys.modules[k]
        try:
            from zonos.config import InferenceParams
        finally:
            sys.path = saved_path

        batch_size, seq_len = hidden_states.shape[:2]
        kv_cache = self.model.backbone.allocate_inference_cache(
            batch_size, seq_len, dtype=hidden_states.dtype
        )
        inference_params = InferenceParams(
            max_seqlen=seq_len,
            max_batch_size=batch_size,
            key_value_memory_dict=kv_cache,
            lengths_per_sample=torch.zeros(batch_size, dtype=torch.long),
        )
        hidden_states = self.model.backbone(
            hidden_states, inference_params=inference_params
        )
        logits = self.model.apply_heads(hidden_states)
        return logits


class ModelVariant(StrEnum):
    """Available Zonos model variants."""

    ZONOS_V0_1_HYBRID = "v0.1-hybrid"


class ModelLoader(ForgeModel):
    """Zonos TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        # Zyphra/Zonos-v0.1-hybrid requires mamba-ssm (CUDA-only). Use the
        # transformer checkpoint which runs on CPU/TT hardware.
        ModelVariant.ZONOS_V0_1_HYBRID: ModelConfig(
            pretrained_model_name="Zyphra/Zonos-v0.1-transformer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ZONOS_V0_1_HYBRID

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._d_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Zonos",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import os
        import sys

        # The local zonos/ model directory (a regular package with __init__.py) shadows
        # the installed zonos pip package (a namespace package without __init__.py).
        # Regular packages beat namespace packages regardless of sys.path order, so we
        # must temporarily remove the shadowing paths before importing.
        worktree_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        saved_path = sys.path[:]
        sys.path = [p for p in sys.path if p not in ("", ".", worktree_dir)]
        for k in [
            k for k in list(sys.modules) if k == "zonos" or k.startswith("zonos.")
        ]:
            del sys.modules[k]
        try:
            from zonos.model import Zonos
        finally:
            sys.path = saved_path

        zonos = Zonos.from_pretrained(
            self._variant_config.pretrained_model_name, device="cpu"
        )
        self._d_model = zonos.config.backbone.d_model

        model = ZonosBackboneWrapper(zonos)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        d_model = self._d_model or 1024
        hidden_states = torch.randn(1, 32, d_model, dtype=dtype)
        return (hidden_states,)
