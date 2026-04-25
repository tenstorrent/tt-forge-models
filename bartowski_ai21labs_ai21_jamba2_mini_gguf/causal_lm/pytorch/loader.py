# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski ai21labs AI21-Jamba2-Mini GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_jamba_gguf():
    """Monkey-patch transformers to add jamba GGUF architecture support.

    Transformers 5.x has JambaForCausalLM but lacks GGUF loading support for
    the jamba architecture. The GGUF metadata stores head_count_kv=0 (only
    Mamba layers have no KV heads), so we skip that field and rely on the
    JambaConfig default (8). All other Mamba-specific fields (mamba_expand,
    attn_layer_period/offset, expert_layer_period/offset) match JambaConfig
    defaults and are not present in the GGUF metadata.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    if "jamba" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # Register jamba as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("jamba")

    # Reuse the mamba tensor processor (handles conv1d reshape and ssm_a log transform)
    if "jamba" not in TENSOR_PROCESSORS and "mamba" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["jamba"] = TENSOR_PROCESSORS["mamba"]

    # Add config mapping for jamba GGUF fields to JambaConfig fields.
    # head_count_kv is 0 in the GGUF (Mamba layers have no KV heads); skip it
    # so JambaConfig defaults to 8. mamba_expand, attn_layer_period/offset, and
    # expert_layer_period/offset are absent from the GGUF but match JambaConfig
    # defaults for Jamba2-Mini.
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["jamba"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": None,
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.state_size": "mamba_d_state",
        "ssm.time_step_rank": "mamba_dt_rank",
        "ssm.inner_size": None,
    }

    # Jamba uses Llama/SentencePiece tokenizer format in GGUF
    if "jamba" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["jamba"] = GGUFLlamaConverter


# Apply the monkey-patch at import time
_patch_transformers_jamba_gguf()


class ModelVariant(StrEnum):
    """Available bartowski ai21labs AI21-Jamba2-Mini GGUF model variants for causal language modeling."""

    AI21_JAMBA2_MINI_Q4_K_M_GGUF = "AI21_Jamba2_Mini_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """bartowski ai21labs AI21-Jamba2-Mini GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AI21_JAMBA2_MINI_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/ai21labs_AI21-Jamba2-Mini-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AI21_JAMBA2_MINI_Q4_K_M_GGUF

    GGUF_FILE = "ai21labs_AI21-Jamba2-Mini-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="bartowski ai21labs AI21-Jamba2-Mini GGUF",
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
        model_kwargs["use_mamba_kernels"] = False
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

        prompts = [self.sample_text]

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
