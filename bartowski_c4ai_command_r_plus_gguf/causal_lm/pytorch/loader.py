# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski c4ai-command-r-plus GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_transformers_command_r_gguf():
    """Monkey-patch transformers to add command-r GGUF architecture support.

    Transformers 5.x has CohereForCausalLM (model_type="cohere") but the GGUF
    format uses "command-r" as the architecture name.  We bridge this by:
    1. Registering "command-r" in GGUF_SUPPORTED_ARCHITECTURES.
    2. Adding the config field mapping from GGUF key names to CohereConfig attrs.
    3. Registering GGUFLlamaConverter for the SentencePiece tokenizer.
    4. Translating model_type "command-r" → "cohere" in load_gguf_checkpoint output
       so that AutoConfig can resolve CohereConfig / CohereForCausalLM.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.configuration_utils as config_utils
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter

    if "command-r" in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        return

    gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("command-r")

    gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["command-r"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "layer_norm_eps",
        "vocab_size": "vocab_size",
    }

    GGUF_TO_FAST_CONVERTERS["command-r"] = GGUFLlamaConverter

    _orig = gguf_utils.load_gguf_checkpoint

    def _patched(*args, **kwargs):
        result = _orig(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "command-r":
            result["config"]["model_type"] = "cohere"
        return result

    gguf_utils.load_gguf_checkpoint = _patched
    config_utils.load_gguf_checkpoint = _patched


_patch_transformers_command_r_gguf()

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
    """Available bartowski c4ai-command-r-plus GGUF model variants for causal language modeling."""

    C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF = "C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """bartowski c4ai-command-r-plus GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/c4ai-command-r-plus-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF

    # Single-file GGUF was replaced by 6 shards; use the first shard (contains full metadata).
    GGUF_FILE = "c4ai-command-r-plus-Q4_K_M.gguf/c4ai-command-r-plus-Q4_K_M-00001-of-00006.gguf"

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
            model="bartowski c4ai-command-r-plus GGUF",
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
