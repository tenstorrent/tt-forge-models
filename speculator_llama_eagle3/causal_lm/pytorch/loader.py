# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeculatorLlama Eagle3 model loader implementation for causal language modeling.
"""
import os

import torch
from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available SpeculatorLlama Eagle3 model variants for causal language modeling."""

    LLAMA3_1_8B_EAGLE3_QUANTIZED = "3.1_8B_Eagle3_Quantized"


TOKENIZER_MODEL_MAP = {
    ModelVariant.LLAMA3_1_8B_EAGLE3_QUANTIZED: "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
}


def _make_random_weight_eagle3(config):
    from speculators.models.eagle3 import Eagle3DraftModel
    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    class _RandomWeightEagle3(Eagle3DraftModel):
        """Eagle3DraftModel subclass that skips verifier weight download
        and uses eager attention (no BlockMask) for TT compiler compatibility."""

        def _setup_embeddings_and_lm_heads(self, config, t2d, embed_requires_grad):
            verifier_config = AutoConfig.from_pretrained(config.name_or_path)
            if hasattr(verifier_config, "text_config"):
                verifier_config = verifier_config.text_config

            self.embed_tokens = torch.nn.Embedding(
                verifier_config.vocab_size,
                self.hidden_size,
                padding_idx=verifier_config.pad_token_id,
            )
            self.lm_head = torch.nn.Linear(
                self.hidden_size, self.draft_vocab_size, bias=False
            )
            self.verifier_lm_head = torch.nn.Linear(
                self.hidden_size, self.draft_vocab_size, bias=False
            )
            self.verifier_norm = LlamaRMSNorm(
                self.hidden_size, eps=verifier_config.rms_norm_eps
            )
            self.verifier_lm_head.weight.requires_grad = False
            self.verifier_norm.weight.requires_grad = False

        def forward(
            self,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            lengths=None,
            loss_mask=None,
            position_ids=None,
            verifier_last_hidden_states=None,
            ttt_steps: int = 3,
            ttt_step_loss_decay: float = 1.0,
            use_off_policy_tokens: bool = False,
            **kwargs,
        ):
            device = hidden_states.device
            total_seq_len = hidden_states.shape[1]

            if position_ids is None:
                position_ids = 1 + torch.arange(
                    total_seq_len, dtype=torch.long, device=device
                ).unsqueeze(0)

            past_key_values = DynamicCache(config=self.config.transformer_layer_config)

            hidden_states = self.fc(hidden_states)

            draft_tokens = []
            for ttt_step in range(ttt_steps):
                with torch.no_grad():
                    input_embeds = self.embed_tokens(input_ids)

                cache_position = torch.arange(
                    ttt_step * total_seq_len,
                    (ttt_step + 1) * total_seq_len,
                    dtype=torch.long,
                    device=device,
                )

                kv_len = (ttt_step + 1) * total_seq_len
                causal_mask = torch.triu(
                    torch.full(
                        (total_seq_len, kv_len),
                        float("-inf"),
                        device=device,
                        dtype=hidden_states.dtype,
                    ),
                    diagonal=1 + ttt_step * total_seq_len,
                )
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

                hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                for decoder_layer in self.layers:
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

                logits = self.lm_head(self.norm(hidden_states))
                input_ids = torch.argmax(logits, dim=-1)
                draft_tokens.append(input_ids.detach().clone())

                if self.d2t is not None:
                    input_ids = input_ids + self.d2t[input_ids]

                position_ids = position_ids + 1

            return draft_tokens

    return _RandomWeightEagle3(config, t2d=None, d2t=None)


class ModelLoader(ForgeModel):
    """SpeculatorLlama Eagle3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.LLAMA3_1_8B_EAGLE3_QUANTIZED: LLMModelConfig(
            pretrained_model_name="nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA3_1_8B_EAGLE3_QUANTIZED

    sample_text = "What is the capital of France?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="speculator_llama_eagle3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        tokenizer_model = TOKENIZER_MODEL_MAP.get(
            self._variant, self._variant_config.pretrained_model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from speculators import SpeculatorModel, SpeculatorModelConfig

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = SpeculatorModelConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config._attn_implementation = "eager"
            if hasattr(config, "transformer_layer_config"):
                config.transformer_layer_config._attn_implementation = "eager"
            model = _make_random_weight_eagle3(config)
        else:
            model = SpeculatorModel.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        if model_kwargs.get("torch_dtype") is not None:
            model = model.to(model_kwargs["torch_dtype"])

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        tokenized = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        input_ids = tokenized["input_ids"]
        seq_len = input_ids.shape[1]
        hidden_size = 4096

        dtype = torch.bfloat16 if dtype_override is None else dtype_override
        hidden_states = torch.randn(1, seq_len, 3 * hidden_size, dtype=dtype)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            hidden_states = hidden_states.repeat_interleave(batch_size, dim=0)

        return {"hidden_states": hidden_states, "input_ids": input_ids}
