# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.2 1B Instruct MLC model loader implementation for causal language modeling.
"""
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
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
    """Available Llama 3.2 1B Instruct MLC model variants for causal language modeling."""

    LLAMA_3_2_1B_INSTRUCT_Q4F16_0_MLC = "Llama_3_2_1B_Instruct_Q4F16_0_MLC"


class ModelLoader(ForgeModel):
    """Llama 3.2 1B Instruct MLC model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_1B_INSTRUCT_Q4F16_0_MLC: LLMModelConfig(
            pretrained_model_name="mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_1B_INSTRUCT_Q4F16_0_MLC

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
            model="Llama 3.2 1B Instruct MLC",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _fix_self_attn_return_count(model):
        """Wrap each decoder layer's self_attn to return the 2-tuple expected by transformers 5.x.

        During torch.compile/XLA tracing the attention forward may return a 3-tuple
        (attn_output, attn_weights, past_key_value). LlamaDecoderLayer.forward in
        transformers 5.x unpacks exactly 2 values, causing 'too many values to unpack'.
        """
        for layer in model.model.layers:
            orig = layer.self_attn.forward

            def _wrap(f):
                def _fwd(*args, **kwargs):
                    result = f(*args, **kwargs)
                    return result[0], result[1]

                return _fwd

            layer.self_attn.forward = _wrap(orig)

    def _load_mlc_chat_config(self):
        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "mlc-chat-config.json"
        )
        with open(config_path) as f:
            return json.load(f)

    def _build_llama_config(self, mlc_config, num_layers=None):
        mc = mlc_config["model_config"]
        rope_scaling = dict(mc["rope_scaling"])
        num_hidden_layers = (
            num_layers if num_layers is not None else mc["num_hidden_layers"]
        )
        return LlamaConfig(
            hidden_size=mc["hidden_size"],
            intermediate_size=mc["intermediate_size"],
            num_attention_heads=mc["num_attention_heads"],
            num_hidden_layers=num_hidden_layers,
            rms_norm_eps=mc["rms_norm_eps"],
            vocab_size=mc["vocab_size"],
            tie_word_embeddings=mc["tie_word_embeddings"],
            rope_theta=mc["position_embedding_base"],
            max_position_embeddings=mc["context_window_size"],
            num_key_value_heads=mc["num_key_value_heads"],
            rope_scaling=rope_scaling,
            bos_token_id=mlc_config["bos_token_id"],
            eos_token_id=mlc_config["eos_token_id"],
            pad_token_id=mlc_config.get("pad_token_id", 0),
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        mlc_config = self._load_mlc_chat_config()
        llama_config = self._build_llama_config(mlc_config, num_layers=self.num_layers)

        # MLC weights are in a compiled binary format incompatible with transformers;
        # build an untrained model with the correct architecture for compilation.
        model = AutoModelForCausalLM.from_config(llama_config).eval()
        self._fix_self_attn_return_count(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = llama_config
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
        mlc_config = self._load_mlc_chat_config()
        self.config = self._build_llama_config(mlc_config)
        return self.config
