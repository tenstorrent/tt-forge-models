# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NV-Embed-v2 model loader implementation for sentence embedding generation.
"""

import torch
from transformers import AutoModel, AutoTokenizer
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
    """Available NV-Embed-v2 model variants for embedding generation."""

    NV_EMBED_V2 = "NV-Embed-v2"


class ModelLoader(ForgeModel):
    """NV-Embed-v2 model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.NV_EMBED_V2: ModelConfig(
            pretrained_model_name="nvidia/NV-Embed-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NV_EMBED_V2

    sample_sentences = [
        "What is the capital of France?",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="NV-Embed-v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    @staticmethod
    def _patch_transformers_compat():
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = (
                lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
            )

        if not hasattr(DynamicCache, "from_legacy_cache"):

            @classmethod
            def _from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                if past_key_values is not None:
                    for layer_idx, (key, value) in enumerate(past_key_values):
                        cache.update(key, value, layer_idx)
                return cache

            DynamicCache.from_legacy_cache = _from_legacy_cache

    @staticmethod
    def _patch_embedding_model_forward(embedding_model):
        from transformers import MistralModel

        embedding_model.forward = MistralModel.forward.__get__(
            embedding_model, type(embedding_model)
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self._patch_transformers_compat()

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        if hasattr(model, "embedding_model"):
            self._patch_embedding_model_forward(model.embedding_model)

        if hasattr(model, "latent_attention_model"):
            self._patch_latent_attention_forward(model.latent_attention_model)

        self._patch_attention_modules(model)
        self._patch_nvembed_forward(model)

        return model

    @staticmethod
    def _patch_latent_attention_forward(latent_attn):
        def patched_latent_forward(self, hiddens, attention_mask=None):
            cross_attn_block, cross_ff = (
                self.cross_attend_blocks[0],
                self.cross_attend_blocks[1],
            )
            b = hiddens.shape[0]
            x = self.latents.unsqueeze(0).expand(b, -1, -1)
            hiddens = cross_attn_block(hiddens, context=x, mask=None) + hiddens
            hiddens = cross_ff(hiddens) + hiddens
            if attention_mask is not None:
                s = torch.sum(hiddens * attention_mask.unsqueeze(-1).float(), dim=1)
                d = attention_mask.sum(dim=1, keepdim=True).float()
                hiddens = s / d
                if self.output_normalize:
                    hiddens = torch.nn.functional.normalize(hiddens, p=2, dim=-1)
            return hiddens

        import types

        latent_attn.forward = types.MethodType(patched_latent_forward, latent_attn)

    @staticmethod
    def _patch_attention_modules(model):
        import types

        def patched_attn_forward(self, x, context=None, mask=None):
            if context is None:
                context = x
            h = self.heads
            q = self.to_q(x)
            kv = self.to_kv(context)
            k, v = kv.chunk(2, dim=-1)
            b, n, _ = q.shape
            d = q.shape[-1] // h
            q = q.view(b, n, h, d).permute(0, 2, 1, 3)
            _, nk, _ = k.shape
            k = k.view(b, nk, h, d).permute(0, 2, 1, 3)
            v = v.view(b, nk, h, d).permute(0, 2, 1, 3)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            out = out.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
            return self.to_out(out)

        for module in model.modules():
            cls_name = type(module).__name__
            if cls_name == "Attention" and hasattr(module, "to_q"):
                module.forward = types.MethodType(patched_attn_forward, module)

    @staticmethod
    def _patch_nvembed_forward(model):
        def patched_forward(
            self, input_ids, attention_mask, pool_mask=None, return_dict=True
        ):
            outputs = self.embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeds = self.latent_attention_model(
                outputs.last_hidden_state,
                pool_mask,
            )
            if not return_dict:
                return (embeds,)
            return {"sentence_embeddings": embeds}

        import types

        model.forward = types.MethodType(patched_forward, model)

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, dict) and "sentence_embeddings" in output:
            return output["sentence_embeddings"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        return token_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, dict) and "sentence_embeddings" in fwd_output:
            return fwd_output["sentence_embeddings"]

        if hasattr(fwd_output, "last_hidden_state"):
            return fwd_output.last_hidden_state

        return fwd_output
