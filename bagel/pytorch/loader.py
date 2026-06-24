# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""BAGEL model loader for tt_forge_models.

BAGEL (ByteDance) is a unified multimodal model shipped as a `trust_remote_code` diffusers
``BagelPipeline``. Repository: https://huggingface.co/JiaxinGe/Diffusers-BAGEL

Components:
  - bagel_model: Qwen2-7B-scale Mixture-of-Transformers (~27 GiB bf16). Weight-bound on a single
    n150/p150 chip -> tensor-parallel (4-way, head-divisible) on the Wormhole fabric.
  - vae: AutoEncoder (~0.31 GiB, single-device).

This loader brings up the **bagel_model** component via a clean, fixed-shape, tensors-only
forward over the Qwen2 MoT backbone (text / "und" understanding path) that lowers to
StableHLO/TTNN. See ``src/bagel_compat.py`` for the trust_remote_code import shims and the
clean-forward wrapper.
"""

from typing import Any, Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.bagel_compat import load_bagel_text_backbone
from .src.shard_specs import get_mesh_shape, shard_backbone_specs


class ModelVariant(StrEnum):
    """Available BAGEL variants."""

    BAGEL_MODEL = "bagel_model"


class ModelLoader(ForgeModel):
    """Loader for the BAGEL bagel_model component (Qwen2 MoT backbone, text path)."""

    _VARIANTS = {
        ModelVariant.BAGEL_MODEL: ModelConfig(
            pretrained_model_name="JiaxinGe/Diffusers-BAGEL",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BAGEL_MODEL

    # Default synthetic-input sequence length for the clean forward.
    DEFAULT_SEQ_LEN = 128
    _VOCAB_SIZE = 151643  # Qwen2 vocab (pad/eos below this)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BAGEL",
            variant=variant,
            group=ModelGroup.PRIORITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the BagelTextBackbone wrapper (clean tensors-only forward)."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return load_bagel_text_backbone(dtype=dtype)

    def load_inputs(self, *, seq_len: Optional[int] = None, **kwargs) -> Any:
        """Synthetic input_ids of shape (1, seq_len) for the text backbone forward."""
        s = seq_len or self.DEFAULT_SEQ_LEN
        generator = torch.Generator().manual_seed(0)
        return torch.randint(
            0, self._VOCAB_SIZE, (1, s), dtype=torch.long, generator=generator
        )

    def get_mesh_config(self, num_devices: int):
        """Megatron-1D mesh for the MoT backbone (4-way preferred; 8-way invalid: 28 heads)."""
        return get_mesh_shape(num_devices)

    def load_shard_spec(self, model):
        """Tensor -> partition-spec dict for the Qwen2 MoT backbone (Megatron-1D)."""
        return shard_backbone_specs(model)

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output
