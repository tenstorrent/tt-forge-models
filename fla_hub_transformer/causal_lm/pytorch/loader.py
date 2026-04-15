# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLA Hub Transformer model loader implementation for causal language modeling.

The fla-hub/transformer models use the Flash Linear Attention (FLA) package's
Transformer architecture, which is a standard multi-head attention transformer
with RoPE and SwiGLU MLP — architecturally identical to LLaMA. Since the FLA
package requires Triton (GPU-only), we load the model as LLaMA with weight key
remapping.
"""
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from typing import Optional
import json

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

# Weight key mapping from FLA Transformer naming to LLaMA naming.
_FLA_TO_LLAMA_KEY_MAP = {
    "model.embeddings.weight": "model.embed_tokens.weight",
    "attn.q_proj.": "self_attn.q_proj.",
    "attn.k_proj.": "self_attn.k_proj.",
    "attn.v_proj.": "self_attn.v_proj.",
    "attn.o_proj.": "self_attn.o_proj.",
    "attn_norm.": "input_layernorm.",
    "mlp_norm.": "post_attention_layernorm.",
}


def _remap_state_dict(state_dict):
    """Remap FLA Transformer weight keys to LLaMA naming convention."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for fla_pattern, llama_pattern in _FLA_TO_LLAMA_KEY_MAP.items():
            new_key = new_key.replace(fla_pattern, llama_pattern)
        new_state_dict[new_key] = value
    return new_state_dict


def _build_llama_config(hub_config, num_layers_override=None):
    """Build a LlamaConfig from the FLA Transformer hub config."""
    num_heads = hub_config["num_heads"]
    num_kv_heads = hub_config.get("num_kv_heads") or num_heads
    hidden_size = hub_config["hidden_size"]
    hidden_ratio = hub_config.get("hidden_ratio", 4)
    intermediate_size = hub_config.get("intermediate_size")
    if intermediate_size is None:
        intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
        # Round up to nearest multiple of 256
        intermediate_size = ((intermediate_size + 255) // 256) * 256

    return LlamaConfig(
        vocab_size=hub_config.get("vocab_size", 32000),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers_override or hub_config["num_hidden_layers"],
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_act="silu",
        max_position_embeddings=hub_config.get("max_position_embeddings", 2048),
        rms_norm_eps=hub_config.get("norm_eps", 1e-6),
        rope_theta=hub_config.get("rope_theta", 10000.0),
        tie_word_embeddings=hub_config.get("tie_word_embeddings", False),
        attention_bias=hub_config.get("attention_bias", False),
        bos_token_id=hub_config.get("bos_token_id", 1),
        eos_token_id=hub_config.get("eos_token_id", 2),
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

    def _load_hub_config(self):
        """Load the raw config.json from the HuggingFace hub."""
        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "config.json"
        )
        with open(config_path) as f:
            return json.load(f)

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        hub_config = self._load_hub_config()
        llama_config = _build_llama_config(hub_config, self.num_layers)

        model = LlamaForCausalLM(llama_config)

        safetensors_path = hf_hub_download(model_name, "model.safetensors")
        fla_state_dict = load_file(safetensors_path)
        llama_state_dict = _remap_state_dict(fla_state_dict)
        model.load_state_dict(llama_state_dict, strict=False)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model = model.eval()
        self.config = model.config
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
        hub_config = self._load_hub_config()
        self.config = _build_llama_config(hub_config, self.num_layers)
        return self.config
