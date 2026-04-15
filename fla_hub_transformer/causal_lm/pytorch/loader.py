# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLA Hub Transformer model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from huggingface_hub import hf_hub_download
import json
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

    def _build_llama_config(self):
        """Convert FLA transformer config to equivalent LlamaConfig."""
        model_name = self._variant_config.pretrained_model_name
        config_path = hf_hub_download(model_name, "config.json")
        with open(config_path) as f:
            fla_config = json.load(f)

        hidden_size = fla_config["hidden_size"]
        hidden_ratio = fla_config.get("hidden_ratio", 4)
        intermediate_size = fla_config.get("intermediate_size")
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

        num_kv_heads = fla_config.get("num_kv_heads")
        if num_kv_heads is None:
            num_kv_heads = fla_config["num_heads"]

        return LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=fla_config["num_hidden_layers"],
            num_attention_heads=fla_config["num_heads"],
            num_key_value_heads=num_kv_heads,
            hidden_act="silu",
            max_position_embeddings=fla_config["max_position_embeddings"],
            rms_norm_eps=fla_config.get("norm_eps", 1e-6),
            vocab_size=fla_config["vocab_size"],
            rope_theta=fla_config.get("rope_theta", 10000.0),
            tie_word_embeddings=fla_config.get("tie_word_embeddings", False),
            bos_token_id=fla_config.get("bos_token_id", 1),
            eos_token_id=fla_config.get("eos_token_id", 2),
            attention_bias=fla_config.get("attention_bias", False),
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        config = self._build_llama_config()
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model = LlamaForCausalLM(config).eval()
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

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
        self.config = self._build_llama_config()
        return self.config
