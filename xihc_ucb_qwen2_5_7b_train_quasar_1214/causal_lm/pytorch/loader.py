# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
xihc-ucb/Qwen2.5-7B-train-Quasar-1214 model loader implementation for causal language modeling.
"""
import sys
import types
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

# ---------------------------------------------------------------------------
# Compatibility shims — applied once at import time.
# ---------------------------------------------------------------------------

# 1. quasar stub: the real quasar package is a proprietary FP8 training library
#    from the model authors and is not publicly available.  We provide the
#    minimum interface needed to load the model for compilation testing.


def _build_quasar_stub() -> None:
    """Inject a minimal quasar stub into sys.modules if quasar is absent."""
    if "quasar" in sys.modules and hasattr(sys.modules["quasar"], "_is_tt_stub"):
        return
    if "quasar" in sys.modules:
        # Real quasar already installed — check it has what we need.
        try:
            from quasar.kernel.configs import QuantType  # noqa: F401

            return
        except ImportError:
            pass  # Fall through to build the stub.

    class QuantType(Enum):
        DIV = "DIV"
        MUL = "MUL"
        SYM = "SYM"
        ASYM = "ASYM"

    @dataclass
    class FP8RMSNormConfig:
        mm_block_size: int = 128
        quant_type: QuantType = QuantType.MUL
        save_fp8_input: bool = False
        scale_dtype: torch.dtype = torch.float32

    @dataclass
    class FP8MulConfig:
        quant_type: QuantType = QuantType.MUL
        scale_dtype: torch.dtype = torch.float32
        float8_dtype: torch.dtype = torch.float8_e4m3fn

    @dataclass
    class FP8DSLinearWithCoatConfig:
        layer_name: str = ""
        scale_dtype: torch.dtype = torch.float32
        fwd_input_quant_type: QuantType = QuantType.DIV
        float8_dtype: torch.dtype = torch.float8_e4m3fn
        act_block_size: int = 16
        mm_block_size: int = 128

    @dataclass
    class FP8QuantConfig:
        float8_dtype: torch.dtype = torch.float8_e4m3fn
        quant_type: QuantType = QuantType.DIV
        fwd_block_size: int = 128
        layer_name: str = ""
        scale_dtype: torch.dtype = torch.float32

    class FP8Identity(nn.Module):
        def forward(self, x):
            return x

    class FP8Quant(nn.Module):
        def __init__(self, quant_config=None):
            super().__init__()

        def forward(self, x):
            return x

    class FP8RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6, norm_config=None):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return self.weight * hidden_states.to(input_dtype)

    class FP8DSLinearWithCoat(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, dsgemm_config=None):
            super().__init__(in_features, out_features, bias=bias)

        def forward(self, x):
            if x.dim() == 3:
                return F.linear(x, self.weight, self.bias)
            return super().forward(x)

    class FP8DSLinearWithCoatWeightBlock(FP8DSLinearWithCoat):
        pass

    class FP8FusedSiLUMul(nn.Module):
        def __init__(self, mul_config=None):
            super().__init__()

        def forward(self, gate, up):
            return F.silu(gate) * up

    # Build module hierarchy
    quasar = types.ModuleType("quasar")
    quasar._is_tt_stub = True

    kernel = types.ModuleType("quasar.kernel")
    quasar.kernel = kernel

    configs_mod = types.ModuleType("quasar.kernel.configs")
    configs_mod.QuantType = QuantType
    configs_mod.FP8RMSNormConfig = FP8RMSNormConfig
    configs_mod.FP8MulConfig = FP8MulConfig
    configs_mod.FP8DSLinearWithCoatConfig = FP8DSLinearWithCoatConfig
    configs_mod.FP8QuantConfig = FP8QuantConfig
    kernel.configs = configs_mod

    quant_pkg = types.ModuleType("quasar.kernel.quant")
    kernel.quant = quant_pkg

    q_mod = types.ModuleType("quasar.kernel.quant.quantize_hp2pb")
    q_mod.fp8_quantize_hp2pb = lambda x, *a, **kw: x
    quant_pkg.quantize_hp2pb = q_mod

    dq_mod = types.ModuleType("quasar.kernel.quant.dequantize_pb2hp")
    dq_mod.fp8_dequantize_pb2hp = lambda x, *a, **kw: x
    quant_pkg.dequantize_pb2hp = dq_mod

    module_pkg = types.ModuleType("quasar.module")
    module_pkg.FP8Identity = FP8Identity
    module_pkg.FP8Quant = FP8Quant
    module_pkg.FP8RMSNorm = FP8RMSNorm
    module_pkg.FP8DSLinearWithCoat = FP8DSLinearWithCoat
    module_pkg.FP8DSLinearWithCoatWeightBlock = FP8DSLinearWithCoatWeightBlock
    module_pkg.FP8FusedSiLUMul = FP8FusedSiLUMul
    quasar.module = module_pkg

    sys.modules["quasar"] = quasar
    sys.modules["quasar.kernel"] = kernel
    sys.modules["quasar.kernel.configs"] = configs_mod
    sys.modules["quasar.kernel.quant"] = quant_pkg
    sys.modules["quasar.kernel.quant.quantize_hp2pb"] = q_mod
    sys.modules["quasar.kernel.quant.dequantize_pb2hp"] = dq_mod
    sys.modules["quasar.module"] = module_pkg


_build_quasar_stub()

# 2. Patch transformers.utils.generic to expose check_model_inputs which was
#    removed in transformers 5.x but referenced by the model's custom code.
import transformers.utils.generic as _tug

if not hasattr(_tug, "check_model_inputs"):
    _tug.check_model_inputs = lambda *args, **kwargs: None


class ModelVariant(StrEnum):
    """Available xihc-ucb/Qwen2.5-7B-train-Quasar-1214 model variants for causal language modeling."""

    XIHC_UCB_QWEN2_5_7B_TRAIN_QUASAR_1214 = "xihc_ucb_qwen2_5_7b_train_quasar_1214"


class ModelLoader(ForgeModel):
    """xihc-ucb/Qwen2.5-7B-train-Quasar-1214 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.XIHC_UCB_QWEN2_5_7B_TRAIN_QUASAR_1214: LLMModelConfig(
            pretrained_model_name="xihc-ucb/Qwen2.5-7B-train-Quasar-1214",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XIHC_UCB_QWEN2_5_7B_TRAIN_QUASAR_1214

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="xihc-ucb/Qwen2.5-7B-train-Quasar-1214",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load config first and pass it directly to from_pretrained to bypass
        # AutoConfig.from_pretrained (which uses return_unused_kwargs=True and
        # triggers a from_dict bug in the model's custom FP8Qwen2Config code).
        if self.config is None:
            self.load_config()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        config = self.config
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
