# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dinerburger Qwen3.5-27B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

# Qwen3.5-27B uses a hybrid SSM + full-attention architecture (qwen35 in GGUF).
# transformers 5.x has native Qwen3_5 support but the GGUF loader doesn't recognise
# the "qwen35" architecture string.  Map it to "qwen3_5_text" so that
# Qwen3_5ForCausalLM is used, which correctly sets q_proj output to
# num_attention_heads * head_dim * 2 (vs Qwen3 which omits the *2 factor).
#
# Other loaders (e.g. bartowski_coniccat) may install a qwen35->qwen3 patch first via
# setdefault.  We use direct assignment to override that mapping, and re-read the GGUF
# architecture header to correct model_type even if an upstream patch already changed
# it to "qwen3".
_QWEN35_CONFIG_MAPPING = {
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
    "attention.key_length": "head_dim",
    "full_attention_interval": "full_attention_interval",
}


def _patch_qwen35_support():
    """Register qwen35 GGUF architecture as Qwen3.5 hybrid (SSM + full-attention).

    Uses direct assignment (not setdefault) so this mapping takes precedence over any
    qwen35->qwen3 alias installed by other loaders (e.g. bartowski_coniccat).
    """
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    # Direct assignment to override any prior setdefault from other loaders
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "qwen35"
    ] = _QWEN35_CONFIG_MAPPING
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _is_qwen35_gguf(gguf_path):
    """Return True if the GGUF file declares architecture 'qwen35'."""
    try:
        from gguf import GGUFReader
        from transformers.modeling_gguf_pytorch_utils import read_field

        reader = GGUFReader(gguf_path)
        arch = read_field(reader, "general.architecture")[0]
        return arch == "qwen35"
    except Exception:
        return False


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to register qwen35 and fix model_type."""
    # Override mapping with direct assignment before any chained patched function
    # can call setdefault (which would be a no-op if already set to the wrong qwen3 mapping).
    _patch_qwen35_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    # Upstream patches (e.g. coniccat) may have changed model_type to "qwen3".
    # Re-read the GGUF architecture directly to correct this.
    if _is_qwen35_gguf(gguf_path):
        result["config"]["model_type"] = "qwen3_5_text"
    return result


_patch_qwen35_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
    """Available dinerburger Qwen3.5-27B GGUF model variants for causal language modeling."""

    QWEN_3_5_27B_IQ4_NL = "27B_IQ4_NL"
    QWEN_3_5_27B_IQ4_XS = "27B_IQ4_XS"
    QWEN_3_5_27B_Q5_K = "27B_Q5_K"
    QWEN_3_5_27B_Q8_0_XXL = "27B_Q8_0_XXL"


class ModelLoader(ForgeModel):
    """dinerburger Qwen3.5-27B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B_IQ4_NL: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_IQ4_XS: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_Q5_K: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_Q8_0_XXL: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B_IQ4_NL

    _GGUF_FILES = {
        ModelVariant.QWEN_3_5_27B_IQ4_NL: "Qwen3.5-27B.IQ4_NL.gguf",
        ModelVariant.QWEN_3_5_27B_IQ4_XS: "Qwen3.5-27B.IQ4_XS.gguf",
        ModelVariant.QWEN_3_5_27B_Q5_K: "Qwen3.5-27B.Q5_K.gguf",
        ModelVariant.QWEN_3_5_27B_Q8_0_XXL: "Qwen3.5-27B.Q8_0_XXL.gguf",
    }

    @property
    def GGUF_FILE(self):
        return self._GGUF_FILES[self._variant]

    sample_text = "Give me a short introduction to large language model."

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
            model="dinerburger Qwen3.5-27B GGUF",
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
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
                config.num_hidden_layers = self.num_layers
                if hasattr(config, "layer_types"):
                    config.layer_types = config.layer_types[: self.num_layers]
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
