# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM


class DeepseekV32Config(DeepseekV3Config):
    model_type = "deepseek_v32"


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    config_class = DeepseekV32Config


try:
    AutoConfig.register("deepseek_v32", DeepseekV32Config)
except ValueError:
    pass
try:
    AutoModelForCausalLM.register(DeepseekV32Config, DeepseekV32ForCausalLM)
except ValueError:
    pass
