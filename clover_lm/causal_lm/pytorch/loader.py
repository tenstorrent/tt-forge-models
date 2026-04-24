# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
daslab-testing/CloverLM model loader implementation for causal language modeling.
"""

import sys
import types
from contextlib import contextmanager
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _unblock(blocked_scales, rows, cols):
    """Reverse cuBLAS blocked layout to row-major for NVFP4 scale factors."""
    n_row_blocks, n_col_blocks = rows // 128, (cols // 16) // 4
    rearranged = blocked_scales.reshape(-1, 32, 4, 4)
    rearranged = rearranged.permute(0, 2, 1, 3).reshape(-1, n_col_blocks, 128, 4)
    rearranged = rearranged.permute(0, 2, 1, 3)
    return rearranged.reshape(n_row_blocks * 128, n_col_blocks * 4)


def _dq_fp4(x_e2m1: torch.Tensor, x_e4m3: torch.Tensor, alpha: float):
    """Dequantize NVFP4 (E2M1) tensors to bfloat16, CPU-compatible."""
    device = x_e2m1.device
    x_e2m1_i32 = x_e2m1.view(dtype=torch.uint8).to(dtype=torch.int32)
    x_e2m1_unpacked = torch.stack(
        [x_e2m1_i32 & 0xF, (x_e2m1_i32 >> 4) & 0xF], dim=-1
    ).flatten(start_dim=-2)
    grid_dq = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    x_fp4_dq = grid_dq[x_e2m1_unpacked]
    scales_dq = x_e4m3.to(torch.float32)
    scales_dq = _unblock(scales_dq, x_e2m1.shape[0], x_e2m1.shape[1] * 2)
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq[..., None]).flatten(
        start_dim=-2
    ) * alpha
    return x_dq.to(torch.bfloat16)


# quartet2 is not on PyPI; provide a CPU-compatible stub so transformers'
# static import checker and the NVFP4 dequantization path both succeed.
if "quartet2" not in sys.modules:
    _quartet2 = types.ModuleType("quartet2")
    _quartet2_linear = types.ModuleType("quartet2.linear")
    _quartet2_linear._dq_fp4 = _dq_fp4
    _quartet2.linear = _quartet2_linear
    sys.modules["quartet2"] = _quartet2
    sys.modules["quartet2.linear"] = _quartet2_linear


@contextmanager
def _cpu_only_cuda():
    """Make tensor.cuda() a no-op so NVFP4 dequantization runs on CPU."""
    original = torch.Tensor.cuda

    def _noop_cuda(self, *args, **kwargs):
        return self

    torch.Tensor.cuda = _noop_cuda
    try:
        yield
    finally:
        torch.Tensor.cuda = original


from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available CloverLM model variants for causal language modeling."""

    CLOVER_LM = "CloverLM"


class ModelLoader(ForgeModel):
    """daslab-testing/CloverLM model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.CLOVER_LM: LLMModelConfig(
            pretrained_model_name="daslab-testing/CloverLM",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CLOVER_LM

    sample_text = "The capital of France is"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CloverLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {
            "trust_remote_code": True,
            # Use plain nn.Linear for inference; FakeQuartetLinear requires CUDA.
            "quartet_2_impl": "bf16",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        with _cpu_only_cuda():
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        return [inputs["input_ids"], inputs["attention_mask"]]
