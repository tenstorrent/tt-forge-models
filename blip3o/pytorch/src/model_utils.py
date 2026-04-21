# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM


class Blip3oQwenConfig(Qwen3Config):
    model_type = "blip3o_qwen"


class Blip3oQwenForCausalLM(Qwen3ForCausalLM):
    config_class = Blip3oQwenConfig


AutoConfig.register("blip3o_qwen", Blip3oQwenConfig)
AutoModelForCausalLM.register(Blip3oQwenConfig, Blip3oQwenForCausalLM)
