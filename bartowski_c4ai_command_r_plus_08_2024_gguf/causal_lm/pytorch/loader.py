# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski c4ai-command-r-plus-08-2024 GGUF model loader implementation for causal language modeling.

Note: The command-r GGUF architecture is not in transformers' default supported
list, so this loader monkey-patches the GGUF utilities to add support for it.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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


def _patch_gguf_command_r_support():
    """Patch transformers GGUF utilities to support the command-r architecture.

    The command-r architecture (used by CohereForAI c4ai-command-r models) is
    stored in GGUF files as architecture "command-r", but transformers 5.2.0
    does not include it in GGUF_SUPPORTED_ARCHITECTURES. The gguf-py library
    already knows the command-r weight name mapping; we only need to add the
    config field mapping, fix the model_type string, and register the tokenizer
    converter (command-r uses a GPT2-style BPE tokenizer).
    """
    import transformers.integrations.ggml as _ggml
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    if "command-r" in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        return

    _ggml.GGUF_CONFIG_MAPPING["command-r"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "layer_norm_eps",
        "logit_scale": "logit_scale",
        "vocab_size": "vocab_size",
    }
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "command-r"
    ] = _ggml.GGUF_CONFIG_MAPPING["command-r"]
    _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES = list(
        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].keys()
    )

    # command-r uses a GPT2-style BPE tokenizer; register under both the
    # GGUF architecture name and the transformers model_type name.
    if "command-r" not in _ggml.GGUF_TO_FAST_CONVERTERS:
        _ggml.GGUF_TO_FAST_CONVERTERS["command-r"] = _ggml.GGUFGPTConverter
    if "cohere" not in _ggml.GGUF_TO_FAST_CONVERTERS:
        _ggml.GGUF_TO_FAST_CONVERTERS["cohere"] = _ggml.GGUFGPTConverter

    _orig_load_gguf = _gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None
    ):
        result = _orig_load_gguf(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )
        if result.get("config", {}).get("model_type") == "command-r":
            result["config"]["model_type"] = "cohere"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf


_patch_gguf_command_r_support()


class ModelVariant(StrEnum):
    """Available bartowski c4ai-command-r-plus-08-2024 GGUF model variants for causal language modeling."""

    BARTOWSKI_C4AI_COMMAND_R_PLUS_08_2024_GGUF = "c4ai_command_r_plus_08_2024_GGUF"


class ModelLoader(ForgeModel):
    """bartowski c4ai-command-r-plus-08-2024 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_C4AI_COMMAND_R_PLUS_08_2024_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/c4ai-command-r-plus-08-2024-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_C4AI_COMMAND_R_PLUS_08_2024_GGUF

    GGUF_FILE = "c4ai-command-r-plus-08-2024-Q2_K.gguf"

    sample_text = "What is the capital of France?"

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
            model="bartowski c4ai-command-r-plus-08-2024 GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load config from GGUF metadata (fast — reads only metadata, no weights).
        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Instantiate on meta device first, then move to CPU with empty (uninitialized)
        # weights. This avoids loading the 37 GB GGUF tensor data, which is too slow
        # and memory-intensive for a 104 B parameter model in a compile-only environment.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        model = model.to_empty(device="cpu")

        self.config = model.config
        self.model = model
        return model.eval()

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
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
