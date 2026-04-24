# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GigaChat3 10B A1.8B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.configuration_utils as _config_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.integrations.ggml import (
    GGUF_TO_FAST_CONVERTERS,
    GGUF_CONFIG_MAPPING,
    GGUFLlamaConverter,
)


def _patch_deepseek_v2_gguf_support():
    """Register deepseek_v2 GGUF architecture support.

    The GigaChat3 model uses the 'deepseek_v2' architecture identifier in its
    GGUF metadata. Transformers 5.x has DeepseekV2ForCausalLM but lacks GGUF
    loading support for the deepseek_v2 architecture. DeepSeek V2 uses a
    SentencePiece BPE tokenizer compatible with GGUFLlamaConverter.

    gguf-py knows this architecture as 'deepseek2', so we also patch
    get_gguf_hf_weights_map to remap deepseek_v2 -> deepseek2 for tensor
    name lookups.
    """
    if "deepseek_v2" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("deepseek_v2")

    GGUF_CONFIG_MAPPING.setdefault(
        "deepseek_v2",
        {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "vocab_size": "vocab_size",
        },
    )

    GGUF_TO_FAST_CONVERTERS.setdefault("deepseek_v2", GGUFLlamaConverter)

    _prev_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "deepseek_v2":
            model_type = "deepseek2"
        return _prev_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_deepseek_v2_gguf_support()

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
    """Available GigaChat3 10B A1.8B GGUF model variants for causal language modeling."""

    GIGACHAT3_10B_A1_8B_GGUF = "10B_A1_8B_GGUF"


class ModelLoader(ForgeModel):
    """GigaChat3 10B A1.8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GIGACHAT3_10B_A1_8B_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GIGACHAT3_10B_A1_8B_GGUF

    GGUF_FILE = "ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="GigaChat3 10B A1.8B GGUF",
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
