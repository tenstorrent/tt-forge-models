# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LFM2 GGUF model loader implementation for causal language modeling.

Supports LiquidAI's LFM2 Mixture-of-Experts models in GGUF format.
"""
from typing import Optional

import torch
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


def _patch_lfm2moe_gguf_support():
    """Patch transformers to add lfm2moe GGUF architecture support.

    The GGUF file uses architecture 'lfm2moe' but transformers only knows 'lfm2'.
    We add 'lfm2moe' as a supported architecture and remap it to the HF model
    type 'lfm2_moe' after loading.
    """
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    import transformers.tokenization_utils_tokenizers as _tok_utils
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        Lfm2TensorProcessor,
        load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    )

    if "lfm2moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["lfm2moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "norm_eps",
        "vocab_size": "vocab_size",
        "shortconv.l_cache": "conv_L_cache",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
    }
    GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")
    TENSOR_PROCESSORS["lfm2moe"] = Lfm2TensorProcessor
    GGUF_TO_FAST_CONVERTERS["lfm2_moe"] = GGUFLlamaConverter

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load_gguf_checkpoint(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "lfm2moe":
            config = result["config"]
            gguf_num_key_value_heads = config.get("num_key_value_heads")
            if isinstance(gguf_num_key_value_heads, list):
                config["num_key_value_heads"] = max(gguf_num_key_value_heads)
                config["block_auto_adjust_ff_dim"] = False
                num_layers = config.get(
                    "num_hidden_layers", len(gguf_num_key_value_heads)
                )
                config["layer_types"] = [
                    "full_attention" if gguf_num_key_value_heads[i] > 0 else "conv"
                    for i in range(num_layers)
                ]
            config["model_type"] = "lfm2_moe"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_lfm2moe_gguf_support()


class ModelVariant(StrEnum):
    """Available LFM2 GGUF model variants for causal language modeling."""

    LFM2_24B_A2B_GGUF = "LFM2_24B_A2B_GGUF"
    MRADERMACHER_LFM2_24B_A2B_GGUF = "mradermacher_LFM2_24B_A2B_GGUF"


class ModelLoader(ForgeModel):
    """LFM2 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LFM2_24B_A2B_GGUF: LLMModelConfig(
            pretrained_model_name="LiquidAI/LFM2-24B-A2B-GGUF",
            max_length=128,
        ),
        ModelVariant.MRADERMACHER_LFM2_24B_A2B_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/LFM2-24B-A2B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_24B_A2B_GGUF

    _GGUF_FILES = {
        ModelVariant.LFM2_24B_A2B_GGUF: "LFM2-24B-A2B-Q4_K_M.gguf",
        ModelVariant.MRADERMACHER_LFM2_24B_A2B_GGUF: "LFM2-24B-A2B.Q4_K_M.gguf",
    }

    sample_text = (
        "What are the key differences between classical and quantum computing?"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LFM2 GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
