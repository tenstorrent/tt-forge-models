# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


class LibraConfig(LlamaConfig):
    model_type = "libra"


class LibraForCausalLM(LlamaForCausalLM):
    config_class = LibraConfig


try:
    AutoConfig.register("libra", LibraConfig)
except ValueError:
    pass
try:
    AutoModelForCausalLM.register(LibraConfig, LibraForCausalLM)
except ValueError:
    pass
