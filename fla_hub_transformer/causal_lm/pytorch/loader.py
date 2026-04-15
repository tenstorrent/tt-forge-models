# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLA Hub Transformer model loader implementation for causal language modeling.

The fla-hub/transformer models use the FLA (Flash Linear Attention) library's
Transformer architecture, which is architecturally identical to Llama (standard
transformer with RoPE, SwiGLU MLP, RMSNorm). Since FLA requires flash-attn
(CUDA-only), we map the config to LlamaForCausalLM for CPU/compile-only
compatibility.
"""
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
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


def _fla_config_to_llama_config(fla_config_dict):
    """Convert an FLA Transformer config dict to a LlamaConfig.

    The FLA Transformer architecture is identical to Llama: standard multi-head
    attention with RoPE, SwiGLU MLP, and RMSNorm pre-normalization.
    """
    hidden_size = fla_config_dict.get("hidden_size", 2048)
    hidden_ratio = fla_config_dict.get("hidden_ratio", 4)
    intermediate_size = fla_config_dict.get("intermediate_size")
    if intermediate_size is None:
        intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
        intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

    num_heads = fla_config_dict.get("num_heads", 32)
    num_kv_heads = fla_config_dict.get("num_kv_heads")
    if num_kv_heads is None:
        num_kv_heads = num_heads

    return LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=fla_config_dict.get("num_hidden_layers", 24),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_act="silu",
        max_position_embeddings=fla_config_dict.get("max_position_embeddings", 2048),
        rms_norm_eps=fla_config_dict.get("norm_eps", 1e-6),
        vocab_size=fla_config_dict.get("vocab_size", 32000),
        rope_theta=fla_config_dict.get("rope_theta", 10000.0),
        tie_word_embeddings=fla_config_dict.get("tie_word_embeddings", False),
        bos_token_id=fla_config_dict.get("bos_token_id", 1),
        eos_token_id=fla_config_dict.get("eos_token_id", 2),
        attention_bias=fla_config_dict.get("attention_bias", False),
    )


class ModelVariant(StrEnum):
    """Available FLA Hub Transformer model variants."""

    TRANSFORMER_1_3B_100B = "1.3B_100B"


class ModelLoader(ForgeModel):
    """FLA Hub Transformer loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.TRANSFORMER_1_3B_100B: LLMModelConfig(
            pretrained_model_name="fla-hub/transformer-1.3B-100B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRANSFORMER_1_3B_100B

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FLA Hub Transformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _get_llama_config(self):
        """Load the FLA config and convert to LlamaConfig."""
        import json

        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "config.json"
        )
        with open(config_path) as f:
            fla_config_dict = json.load(f)

        llama_config = _fla_config_to_llama_config(fla_config_dict)
        if self.num_layers is not None:
            llama_config.num_hidden_layers = self.num_layers
        return llama_config

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        llama_config = self._get_llama_config()
        if dtype_override is not None:
            llama_config.torch_dtype = dtype_override

        model = LlamaForCausalLM(llama_config).eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        self.config = llama_config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def load_config(self):
        self.config = self._get_llama_config()
        return self.config
