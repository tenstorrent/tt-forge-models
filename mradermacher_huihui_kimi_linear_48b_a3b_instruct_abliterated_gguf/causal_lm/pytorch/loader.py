# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF model loader for causal language modeling.

Note: The kimi-linear GGUF architecture is not supported by the transformers GGUF
loader. Instead this loader uses the original model code (moonshotai/Kimi-Linear-48B-A3B-Instruct)
with local patches to fix import incompatibilities, and initialises the model from
config with a reduced layer count for compile-only bringup testing.
"""

import os

# Disable hf-xet download backend to prevent hangs on large model downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

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

# Base model repo for config/tokenizer (the abliterated GGUF weights cannot be
# loaded via transformers because kimi-linear GGUF architecture is unsupported).
_BASE_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

# Load local patched model code (fixes OutputRecorder import, fla stubs, and
# removes the hard-coded flash_attention_2 forcing).
_SRC_DIR = Path(__file__).parent / "src"


def _load_kimi_modules():
    conf_path = _SRC_DIR / "configuration_kimi.py"
    model_path = _SRC_DIR / "modeling_kimi.py"

    conf_spec = importlib.util.spec_from_file_location("configuration_kimi", conf_path)
    conf_mod = importlib.util.module_from_spec(conf_spec)
    sys.modules["configuration_kimi"] = conf_mod
    conf_spec.loader.exec_module(conf_mod)

    model_spec = importlib.util.spec_from_file_location("modeling_kimi", model_path)
    model_mod = importlib.util.module_from_spec(model_spec)
    sys.modules["modeling_kimi"] = model_mod
    model_spec.loader.exec_module(model_mod)

    return conf_mod.KimiLinearConfig, model_mod.KimiLinearForCausalLM


class ModelVariant(StrEnum):
    """Available mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF model variants."""

    HUIHUI_KIMI_LINEAR_48B_A3B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF = (
        "48B_A3B_Instruct_Abliterated_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.HUIHUI_KIMI_LINEAR_48B_A3B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.HUIHUI_KIMI_LINEAR_48B_A3B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF
    )

    # Number of transformer layers to use when creating the model from config.
    # The full 48B model has 27 layers; a small value keeps memory manageable
    # for compile-only bringup testing.
    DEFAULT_NUM_LAYERS = 2

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = (
            num_layers if num_layers is not None else self.DEFAULT_NUM_LAYERS
        )

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            _BASE_MODEL, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        KimiLinearConfig, KimiLinearForCausalLM = _load_kimi_modules()

        config = KimiLinearConfig.from_pretrained(_BASE_MODEL)
        config.num_hidden_layers = self.num_layers
        # Disable KDA (linear/SSM attention) layers: the fla-core kernels require
        # CUDA/triton which is not available in this environment.  Setting kda_layers
        # to empty forces all layers to use standard MLA attention instead.
        config.linear_attn_config["kda_layers"] = []
        config._attn_implementation = "eager"

        torch_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = KimiLinearForCausalLM(config).to(torch_dtype).eval()

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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            if hasattr(layer, "block_sparse_moe"):
                moe = layer.block_sparse_moe
                if hasattr(moe, "experts"):
                    shard_specs[moe.experts.gate_up_proj] = (None, "model", "batch")
                    shard_specs[moe.experts.down_proj] = (None, "batch", "model")
            elif hasattr(layer, "mlp"):
                mlp = layer.mlp
                if hasattr(mlp, "gate_proj"):
                    shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                    shard_specs[mlp.up_proj.weight] = ("model", "batch")
                    shard_specs[mlp.down_proj.weight] = ("batch", "model")
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                if hasattr(attn, "q_proj"):
                    shard_specs[attn.q_proj.weight] = ("model", "batch")
                if hasattr(attn, "o_proj"):
                    shard_specs[attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        KimiLinearConfig, _ = _load_kimi_modules()
        config = KimiLinearConfig.from_pretrained(_BASE_MODEL)
        config.num_hidden_layers = self.num_layers
        config.linear_attn_config["kda_layers"] = []
        config._attn_implementation = "eager"
        self.config = config
        return self.config
