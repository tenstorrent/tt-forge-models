# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE Reranker v2.5 Gemma2 Lightweight model loader implementation for passage ranking.
"""
import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available BGE Reranker v2.5 Gemma2 Lightweight model variants for passage ranking."""

    LIGHTWEIGHT = "lightweight"


class ModelLoader(ForgeModel):
    """BGE Reranker v2.5 Gemma2 Lightweight model loader implementation for passage ranking.

    This reranker uses a Gemma2-based causal LM backbone to score query-passage
    relevance. It formats input as "A: {query}\\nB: {passage}\\nPredict whether
    passage B contains an answer to query A." and uses the last token logit as
    the relevance score.
    """

    _VARIANTS = {
        ModelVariant.LIGHTWEIGHT: ModelConfig(
            pretrained_model_name="BAAI/bge-reranker-v2.5-gemma2-lightweight",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LIGHTWEIGHT

    # Sample query-passage pairs for testing
    sample_pairs = [
        ("what is panda?", "hi"),
        (
            "what is panda?",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        ),
    ]

    _PROMPT_TEMPLATE = "A: {query}\nB: {passage}\nPredict whether passage B contains an answer to query A."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BGE-Reranker-v2.5-Gemma2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        import transformers.models.gemma2.modeling_gemma2 as _gemma2_mod
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.models.gemma2.modeling_gemma2 import (
            Gemma2Attention,
            Gemma2DecoderLayer,
            Gemma2RotaryEmbedding,
        )

        # transformers 5.x removed the per-backend attention subclasses and docstring
        # constants; the custom remote code for this model still imports them.
        for _name, _val in [
            ("Gemma2FlashAttention2", Gemma2Attention),
            ("Gemma2SdpaAttention", Gemma2Attention),
            ("GEMMA2_ATTENTION_CLASSES", {"eager": Gemma2Attention, "flash_attention_2": Gemma2Attention, "sdpa": Gemma2Attention}),
            ("GEMMA2_START_DOCSTRING", ""),
            ("GEMMA2_INPUTS_DOCSTRING", ""),
        ]:
            if not hasattr(_gemma2_mod, _name):
                setattr(_gemma2_mod, _name, _val)

        # transformers 5.x changed _tied_weights_keys from list to dict; pre-import
        # the dynamic class and fix the class attribute before from_pretrained
        # instantiates it. When layer_wise=True, lm_head is a ModuleList with no
        # .weight, so weight tying must be disabled for that variant.
        _config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        _cls = get_class_from_dynamic_module(
            "gemma_model.CostWiseGemmaForCausalLM",
            self._variant_config.pretrained_model_name,
        )
        if isinstance(_cls._tied_weights_keys, list):
            if _config.layer_wise:
                _cls._tied_weights_keys = {}
            else:
                _cls._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

        # transformers 5.x moved RoPE computation out of Gemma2Attention into the
        # parent Gemma2Model.forward, so Gemma2DecoderLayer now expects a
        # position_embeddings tuple.  The custom CostWiseGemmaModel was written for
        # the old API and does not pass position_embeddings.  Patch the decoder layer
        # to compute them lazily when not provided (guard: only when None, so native
        # Gemma2Model usage is unaffected).
        if not getattr(Gemma2DecoderLayer, "_compat_position_emb_patched", False):
            _orig_decoder_fwd = Gemma2DecoderLayer.forward

            def _compat_decoder_fwd(
                self,
                hidden_states,
                position_embeddings=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                cache_position=None,
                **_kwargs,
            ):
                if position_embeddings is None and position_ids is not None:
                    if not hasattr(self, "_compat_rotary_emb"):
                        self._compat_rotary_emb = Gemma2RotaryEmbedding(self.config)
                    self._compat_rotary_emb = self._compat_rotary_emb.to(
                        hidden_states.device
                    )
                    position_embeddings = self._compat_rotary_emb(
                        hidden_states, position_ids
                    )
                result = _orig_decoder_fwd(
                    self,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **_kwargs,
                )
                # transformers 5.x returns plain hidden_states tensor; old custom
                # model code does layer_outputs[0] expecting a tuple return.
                if isinstance(result, torch.Tensor):
                    return (result,)
                return result

            Gemma2DecoderLayer.forward = _compat_decoder_fwd
            Gemma2DecoderLayer._compat_position_emb_patched = True

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        texts = [
            self._PROMPT_TEMPLATE.format(query=query, passage=passage)
            for query, passage in self.sample_pairs
        ]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
