# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski ibm-granite granite-4.0-micro GGUF model loader implementation for causal language modeling.
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


def _patch_granite_gguf_support():
    """Register granite GGUF architecture support for GraniteMoeHybrid models.

    IBM Granite 4.0 Micro is a hybrid (attention + Mamba SSM) model. The GGUF
    file reports architecture 'granite' but transformers 5.x only knows this as
    'granitemoehybrid'. This patch bridges the gap by registering the config
    field mappings and tokenizer converter.
    """
    if "granite" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("granite")

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["granite"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.scale": "attention_multiplier",
        "embedding_scale": "embedding_multiplier",
        "logit_scale": "logits_scaling",
        "residual_scale": "residual_multiplier",
        "expert_count": "num_local_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_feed_forward_length": "shared_intermediate_size",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.group_count": "mamba_n_groups",
        "ssm.state_size": "mamba_d_state",
        "vocab_size": "vocab_size",
    }

    # Granite uses a GPT-2 BPE tokenizer. Register both architecture names so
    # convert_gguf_tokenizer works whether model_type is 'granite' or 'granitemoehybrid'.
    if "gpt2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("granite", GGUF_TO_FAST_CONVERTERS["gpt2"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "granitemoehybrid", GGUF_TO_FAST_CONVERTERS["gpt2"]
        )


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add granite GGUF support."""
    import transformers.utils.import_utils as _import_utils

    if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
    _patch_granite_gguf_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    config = result.get("config", {})
    if config.get("model_type") == "granite":
        # num_key_value_heads is stored as a per-layer array in granite GGUF;
        # extract the single scalar value that GraniteConfig expects.
        kv_heads = config.get("num_key_value_heads")
        if isinstance(kv_heads, list) and len(kv_heads) > 0:
            config["num_key_value_heads"] = kv_heads[0]
    return result


_patch_granite_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available bartowski ibm-granite granite-4.0-micro GGUF model variants for causal language modeling."""

    IBM_GRANITE_GRANITE_4_0_MICRO_Q4_K_M_GGUF = (
        "IBM_Granite_Granite_4_0_Micro_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """bartowski ibm-granite granite-4.0-micro GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.IBM_GRANITE_GRANITE_4_0_MICRO_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/ibm-granite_granite-4.0-micro-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IBM_GRANITE_GRANITE_4_0_MICRO_Q4_K_M_GGUF

    GGUF_FILE = "ibm-granite_granite-4.0-micro-Q4_K_M.gguf"

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
            model="bartowski ibm-granite granite-4.0-micro GGUF",
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
