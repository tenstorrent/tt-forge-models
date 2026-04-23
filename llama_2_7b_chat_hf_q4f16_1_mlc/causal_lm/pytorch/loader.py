# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 2 7B Chat HF Q4F16_1 MLC model loader implementation for causal language modeling.
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


class ModelVariant(StrEnum):
    """Available Llama 2 7B Chat HF Q4F16_1 MLC model variants."""

    LLAMA_2_7B_CHAT_HF_Q4F16_1_MLC = "llama_2_7b_chat_hf_q4f16_1_mlc"


class ModelLoader(ForgeModel):
    """Llama 2 7B Chat HF Q4F16_1 MLC model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LLAMA_2_7B_CHAT_HF_Q4F16_1_MLC: LLMModelConfig(
            pretrained_model_name="mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_2_7B_CHAT_HF_Q4F16_1_MLC

    # MLC model uses the same tokenizer as the base Llama-2-7b-chat-hf model.
    # NousResearch hosts a public, ungated mirror of the tokenizer.
    _TOKENIZER_SOURCE = "NousResearch/Llama-2-7b-chat-hf"

    sample_text = "Hey how are you doing today?"

    # Architecture parameters sourced from mlc-chat-config.json in the model repo.
    # The MLC repo stores weights in a custom shard format incompatible with
    # standard transformers, so we always build the model from this config with
    # random weights.
    _LLAMA_CONFIG = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        rms_norm_eps=1e-06,
        vocab_size=32000,
        max_position_embeddings=4096,
        num_key_value_heads=32,
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Llama 2 7B Chat HF Q4F16_1 MLC",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_SOURCE)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = LlamaConfig(
            hidden_size=self._LLAMA_CONFIG.hidden_size,
            intermediate_size=self._LLAMA_CONFIG.intermediate_size,
            num_attention_heads=self._LLAMA_CONFIG.num_attention_heads,
            num_hidden_layers=(
                self.num_layers
                if self.num_layers is not None
                else self._LLAMA_CONFIG.num_hidden_layers
            ),
            rms_norm_eps=self._LLAMA_CONFIG.rms_norm_eps,
            vocab_size=self._LLAMA_CONFIG.vocab_size,
            max_position_embeddings=self._LLAMA_CONFIG.max_position_embeddings,
            num_key_value_heads=self._LLAMA_CONFIG.num_key_value_heads,
        )

        model = LlamaForCausalLM(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()

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

        sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

        if batch_size > 1:
            for i in range(len(sample_inputs)):
                sample_inputs[i] = sample_inputs[i].repeat_interleave(batch_size, dim=0)

        return sample_inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            decoded_output = self.tokenizer.decode(outputs)
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_output = self.tokenizer.decode(next_token_id)

        return decoded_output

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = LlamaConfig(
            hidden_size=self._LLAMA_CONFIG.hidden_size,
            intermediate_size=self._LLAMA_CONFIG.intermediate_size,
            num_attention_heads=self._LLAMA_CONFIG.num_attention_heads,
            num_hidden_layers=self._LLAMA_CONFIG.num_hidden_layers,
            rms_norm_eps=self._LLAMA_CONFIG.rms_norm_eps,
            vocab_size=self._LLAMA_CONFIG.vocab_size,
            max_position_embeddings=self._LLAMA_CONFIG.max_position_embeddings,
            num_key_value_heads=self._LLAMA_CONFIG.num_key_value_heads,
        )
        return self.config
