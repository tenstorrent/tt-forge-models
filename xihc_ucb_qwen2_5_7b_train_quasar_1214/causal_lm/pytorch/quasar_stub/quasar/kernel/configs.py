# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stub quasar.kernel.configs for FP8 quantization config types."""
import torch
from dataclasses import dataclass, field
from enum import Enum


class QuantType(Enum):
    DIV = "DIV"
    MUL = "MUL"


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


@dataclass
class FP8DSLinearWithCoatConfig:
    layer_name: str = ""
    scale_dtype: torch.dtype = torch.float32
    fwd_input_quant_type: QuantType = QuantType.DIV


@dataclass
class FP8QuantConfig:
    float8_dtype: torch.dtype = torch.float8_e4m3fn
    quant_type: QuantType = QuantType.DIV
    fwd_block_size: int = 128
    layer_name: str = ""
    scale_dtype: torch.dtype = torch.float32
