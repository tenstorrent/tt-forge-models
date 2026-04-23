# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternLM2-Math-Plus 7B GGUF model loader implementation for causal language modeling.
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


def _patch_internlm2_gguf_support():
    """Register internlm2 as a llama-compatible GGUF architecture.

    Transformers 5.x does not recognise 'internlm2' as a GGUF architecture.
    InternLM2 uses the same standardised GGUF tensor names as LLaMA
    (token_embd, blk.N.attn_q, etc.), so we register it as an alias for
    llama and remap model_type in the loaded config.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.tokenization_utils_tokenizers as tok_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        load_gguf_checkpoint as _orig_load,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    if "internlm2" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("internlm2")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["internlm2"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    for arch in ("internlm2",):
        GGUF_TO_FAST_CONVERTERS.setdefault(arch, GGUFLlamaConverter)

    def _patched_load(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "internlm2":
            result["config"]["model_type"] = "llama"
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load
    for mod in (config_utils, modeling_utils, tok_auto, tok_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _patched_load


_patch_internlm2_gguf_support()


class ModelVariant(StrEnum):
    """Available InternLM2-Math-Plus 7B GGUF model variants for causal language modeling."""

    INTERNLM2_MATH_PLUS_7B_Q4_K_M = "7B_Math_Plus_Q4_K_M"


class ModelLoader(ForgeModel):
    """InternLM2-Math-Plus 7B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.INTERNLM2_MATH_PLUS_7B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="RichardErkhov/internlm_-_internlm2-math-plus-7b-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERNLM2_MATH_PLUS_7B_Q4_K_M

    GGUF_FILE = "internlm2-math-plus-7b.Q4_K_M.gguf"

    sample_text = "Solve the equation: 2x + 5 = 13."

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
            model="InternLM2-Math-Plus 7B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
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

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                gguf_file=self.GGUF_FILE,
                trust_remote_code=True,
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

        inputs = self.tokenizer(
            self.sample_text,
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
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
            trust_remote_code=True,
        )
        return self.config
