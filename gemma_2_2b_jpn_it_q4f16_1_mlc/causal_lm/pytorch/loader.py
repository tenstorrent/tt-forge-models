# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 2 2B JPN IT Q4F16_1 MLC model loader implementation for causal language modeling.
"""
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config
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
    """Available Gemma 2 2B JPN IT Q4F16_1 MLC model variants for causal language modeling."""

    GEMMA_2_2B_JPN_IT_Q4F16_1_MLC = "Gemma_2_2B_JPN_IT_Q4F16_1_MLC"


class ModelLoader(ForgeModel):
    """Gemma 2 2B JPN IT Q4F16_1 MLC model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_2_2B_JPN_IT_Q4F16_1_MLC: LLMModelConfig(
            pretrained_model_name="mlc-ai/gemma-2-2b-jpn-it-q4f16_1-MLC",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_2_2B_JPN_IT_Q4F16_1_MLC

    sample_text = "富士山の高さは何メートルですか？"

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
            model="Gemma 2 2B JPN IT Q4F16_1 MLC",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_config_from_mlc(self, num_layers=None):
        """Build a Gemma2Config from mlc-chat-config.json since the repo lacks config.json."""
        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "mlc-chat-config.json"
        )
        with open(config_path) as f:
            mlc_config = json.load(f)

        mc = mlc_config["model_config"]
        n_layers = num_layers if num_layers is not None else mc["num_hidden_layers"]
        return Gemma2Config(
            vocab_size=mlc_config.get("vocab_size", 256000),
            hidden_size=mc["hidden_size"],
            intermediate_size=mc["intermediate_size"],
            num_hidden_layers=n_layers,
            num_attention_heads=mc["num_attention_heads"],
            num_key_value_heads=mc["num_key_value_heads"],
            head_dim=mc["head_dim"],
            hidden_activation=mc.get("hidden_activation", "gelu_pytorch_tanh"),
            max_position_embeddings=mc.get("context_window_size", 4096),
            rms_norm_eps=mc["rms_norm_eps"],
            attention_bias=mc.get("attention_bias", False),
            attn_logit_softcapping=mc.get("attn_logit_softcapping", 50.0),
            final_logit_softcapping=mc.get("final_logit_softcapping", 30.0),
            query_pre_attn_scalar=mc.get("query_pre_attn_scalar", 224),
            sliding_window=mc.get("sliding_window", 4096),
            pad_token_id=mlc_config.get("pad_token_id", 0),
            eos_token_id=mlc_config.get("eos_token_id", 1),
            bos_token_id=mlc_config.get("bos_token_id", 2),
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

        config = self._build_config_from_mlc(num_layers=self.num_layers)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs).eval()

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
        self.config = self._build_config_from_mlc()
        return self.config
