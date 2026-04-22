# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Evo 2 model loader implementation for causal language modeling.

Evo 2 uses the StripedHyena2 architecture (via the `vtx` / vortex package).
The vortex package imports flash_attn_2_cuda and triton at module level, which
are NVIDIA CUDA extensions. We mock these at import time and replace the
triton-based rotary kernel with a pure-PyTorch equivalent so the model can be
instantiated and traced on non-CUDA hardware (e.g. Tenstorrent).
"""

import sys
import types

import torch
from einops import repeat

# ---------------------------------------------------------------------------
# 1.  CUDA / GPU mock setup – must happen before any vortex imports.
# ---------------------------------------------------------------------------

if "flash_attn_2_cuda" not in sys.modules:
    sys.modules["flash_attn_2_cuda"] = types.ModuleType("flash_attn_2_cuda")

if "triton" not in sys.modules:

    class _MockConstexpr:
        pass

    _mock_tl = types.ModuleType("triton.language")
    for _name in [
        "constexpr",
        "float32",
        "float16",
        "bfloat16",
        "int32",
        "int64",
        "bool",
        "int8",
        "uint8",
        "dtype",
    ]:
        setattr(_mock_tl, _name, _MockConstexpr())
    _mock_tl.program_id = lambda axis: 0

    _mock_triton = types.ModuleType("triton")
    _mock_triton.jit = lambda fn=None, **k: (fn if fn else lambda f: f)
    _mock_triton.cdiv = lambda x, y: (x + y - 1) // y
    _mock_triton.language = _mock_tl

    sys.modules["triton"] = _mock_triton
    sys.modules["triton.language"] = _mock_tl


# ---------------------------------------------------------------------------
# 2.  Import vortex rotary module then replace the triton kernel with a
#     pure-PyTorch implementation.
# ---------------------------------------------------------------------------

import vortex.ops.embedding.rotary as _rotary_mod  # noqa: E402


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_cpu(
    x,
    cos,
    sin,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    interleaved=False,
    inplace=False,
    conjugate=False,
):
    ro_dim = cos.shape[-1] * 2
    pattern = "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    cos_ = repeat(cos, pattern)
    sin_ = repeat(sin, pattern)
    if conjugate:
        sin_ = -sin_
    x_rot = x[..., :ro_dim] * cos_ + _rotate_half(x[..., :ro_dim]) * sin_
    result = (
        torch.cat([x_rot, x[..., ro_dim:]], dim=-1) if ro_dim < x.shape[-1] else x_rot
    )
    if inplace:
        x[...] = result
        return x
    return result


_rotary_mod.apply_rotary = _apply_rotary_cpu

# ---------------------------------------------------------------------------
# 3.  Patch torch.cuda.device so the vortex model initialiser (which always
#     uses `with torch.cuda.device(device)` even on CPU) doesn't crash.
# ---------------------------------------------------------------------------


class _NoopCudaDevice:
    def __init__(self, device=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


torch.cuda.device = _NoopCudaDevice  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 4.  Now it is safe to import the actual vortex / evo2 model classes.
# ---------------------------------------------------------------------------

import importlib.util
import pkgutil
import yaml

from vortex.model.model import StripedHyena  # noqa: E402
from vortex.model.utils import dotdict  # noqa: E402

# CharLevelTokenizer lives in vortex but importing vortex.model.tokenizer is
# safe (no CUDA deps in that file).
from vortex.model.tokenizer import CharLevelTokenizer  # noqa: E402

from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Evo 2 model variants for causal language modeling."""

    EVO2_7B = "7b"


# ---------------------------------------------------------------------------
# 5.  Thin nn.Module wrapper so the test harness gets a standard interface.
# ---------------------------------------------------------------------------


class _Evo2Wrapper(torch.nn.Module):
    """Wrap StripedHyena to expose a (input_ids) -> logits interface."""

    def __init__(self, backbone: StripedHyena):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.backbone(input_ids)
        return logits


# ---------------------------------------------------------------------------
# 6.  ForgeModel loader
# ---------------------------------------------------------------------------


class ModelLoader(ForgeModel):
    """Evo 2 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EVO2_7B: LLMModelConfig(
            pretrained_model_name="arcinstitute/evo2_7b",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EVO2_7B

    # Character-level DNA/RNA input
    sample_text = "ACGTACGTACGTACGTACGTACGTACGTACGT"

    # Vocab size matches CharLevelTokenizer (512 byte-level tokens)
    _VOCAB_SIZE = 512

    # Config bundled with the evo2 PyPI package
    _EVO2_CONFIG_NAME = "evo2-7b-8k.yml"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer: Optional[CharLevelTokenizer] = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Evo2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = CharLevelTokenizer(self._VOCAB_SIZE)
        return self.tokenizer

    def _load_evo2_config(self) -> dotdict:
        """Load the evo2 7b config bundled in the evo2 PyPI package."""
        spec = importlib.util.find_spec("evo2")
        if spec is None:
            raise ImportError("evo2 package not installed – run: pip install evo2")

        # The installed package may be shadowed by the local evo2/ model directory.
        # Walk the search paths to find the installed package.
        evo2_install_dirs = []
        if spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                if "site-packages" in loc:
                    evo2_install_dirs.append(loc)

        if not evo2_install_dirs:
            # Fallback: locate via pkgutil in site-packages
            import site

            for sp in site.getsitepackages():
                candidate = __import__("os").path.join(sp, "evo2")
                if __import__("os").path.isdir(candidate):
                    evo2_install_dirs.append(candidate)

        if not evo2_install_dirs:
            raise ImportError("Could not locate installed evo2 package configs")

        import os

        config_path = os.path.join(
            evo2_install_dirs[0], "configs", self._EVO2_CONFIG_NAME
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        config = dotdict(config)
        # Disable CUDA-specific options for CPU / TT hardware compatibility.
        config.use_fp8_input_projections = False
        config.use_flash_attn = False

        if self.num_layers is not None:
            config.num_layers = self.num_layers

        return config

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Evo 2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype (default: torch.bfloat16).

        Returns:
            torch.nn.Module: Wrapped StripedHyena model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        config = self._load_evo2_config()
        backbone = StripedHyena(config)
        backbone = backbone.to(dtype)
        backbone.eval()

        model = _Evo2Wrapper(backbone)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Evo 2 model.

        Args:
            dtype_override: Unused (inputs are integer token IDs).
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length
        prompts = [self.sample_text] * batch_size
        ids_list = [self.tokenizer.tokenize(p) for p in prompts]

        # Pad / truncate to max_length
        padded = []
        for ids in ids_list:
            ids = list(ids)[:max_length]
            ids += [self.tokenizer.pad_id] * (max_length - len(ids))
            padded.append(ids)

        input_ids = torch.tensor(padded, dtype=torch.long)
        return {"input_ids": input_ids}
