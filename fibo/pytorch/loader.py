# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO (briaai/FIBO) model loader implementation.

FIBO is BRIA AI's ~8B-parameter DiT-based flow-matching text-to-image model.
It uses SmolLM3-3B as the text encoder, a Wan VAE, and a Flux-style MMDiT
denoiser (``BriaFiboTransformer2DModel``) with a "DimFusion" conditioning
architecture (paper: arXiv:2511.06876).

Diffusion pipelines are brought up by composite component, not as one graph.
This loader targets the **denoiser** — the compute-dominant ``BriaFiboTransformer2DModel``
that must run on device. ``load_model`` loads only the transformer (the 3B text
encoder and VAE are not needed for the denoiser forward, keeping host memory
bounded), and ``load_inputs`` builds synthetic inputs at FIBO's native
1024x1024 resolution (latent sequence length 4096).

The denoiser alone is ~16.5 GB in bf16 and does not fit one Wormhole chip
(12 GB), so on multi-chip parts it is sharded tensor-parallel; ``get_mesh_config``
and ``load_shard_spec`` provide a Megatron-style layout (see
``src/model_utils.fibo_shard_specs``).

Reference: https://huggingface.co/briaai/FIBO
"""

from typing import Optional

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
from .src.model_utils import (
    apply_single_block_rewrite,
    build_fibo_inputs,
    fibo_shard_specs,
)


class ModelVariant(StrEnum):
    """Available FIBO model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """FIBO denoiser (``BriaFiboTransformer2DModel``) loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="briaai/FIBO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given FIBO variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIBO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the FIBO denoiser (transformer) module.

        Args:
            dtype_override: Optional ``torch.dtype``. Defaults to ``bfloat16``
                (the checkpoint's native format and the device test dtype).

        Returns:
            torch.nn.Module: ``BriaFiboTransformer2DModel`` in eval mode.
        """
        from diffusers import BriaFiboTransformer2DModel

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = BriaFiboTransformer2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        # Split each single block's fused proj_out into attn/mlp halves to avoid
        # a 15360-wide concat that exceeds Wormhole L1 (numerically identical).
        apply_single_block_rewrite(model)
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return synthetic denoiser inputs at FIBO's native 1024x1024 resolution.

        Args:
            dtype_override: Optional ``torch.dtype`` for the float tensors.
            batch_size: Base batch size (default 1 — a single denoising forward).

        Returns:
            dict: kwargs for ``BriaFiboTransformer2DModel.forward`` (the runner
            invokes ``model(**inputs)``).
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return build_fibo_inputs(dtype=dtype, batch_size=batch_size)

    def unpack_forward_output(self, fwd_output):
        """Return the predicted-noise ``sample`` tensor from the denoiser output.

        With ``return_dict=False`` the transformer returns a ``(sample,)`` tuple.
        """
        if isinstance(fwd_output, (tuple, list)):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output

    # --- Tensor-parallel support (denoiser is too large for a single chip) ---

    def get_mesh_config(self, num_devices: int):
        """1xN tensor-parallel mesh over ``num_devices`` (e.g. (1, 2) on n300)."""
        return (1, num_devices), ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP shard specs for the FIBO denoiser weights."""
        return fibo_shard_specs(model, model_axis="model")
