# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VinAI PhoGPT-4B-Chat causal language model loader implementation.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# transformers >=5.x removed LlamaLinearScalingRotaryEmbedding and
# LlamaDynamicNTKScalingRotaryEmbedding; provide stubs so modeling_mpt.py imports succeed.
import transformers.models.llama.modeling_llama as _llama_module

if not hasattr(_llama_module, "LlamaDynamicNTKScalingRotaryEmbedding"):

    class _LlamaCompatRotaryEmbedding(nn.Module):
        def __init__(
            self,
            dim,
            max_position_embeddings=2048,
            base=10000,
            scaling_factor=1.0,
            device=None,
        ):
            super().__init__()
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            self.scaling_factor = scaling_factor
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, x, seq_len=None):
            seq_len = seq_len or x.shape[-2]
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    _llama_module.LlamaDynamicNTKScalingRotaryEmbedding = _LlamaCompatRotaryEmbedding
    _llama_module.LlamaLinearScalingRotaryEmbedding = _LlamaCompatRotaryEmbedding


def _patch_mpt_post_init() -> None:
    """Patch MPTForCausalLM for transformers >=5.x compatibility.

    1. MPT models written for older transformers omit self.post_init() at the
       end of __init__, which is now required to populate all_tied_weights_keys.
    2. tie_weights() must accept **kwargs (transformers 5.x passes recompute_mapping).
    """
    import sys

    from transformers import PreTrainedModel

    for module in list(sys.modules.values()):
        cls = getattr(module, "MPTForCausalLM", None)
        if (
            cls is not None
            and isinstance(cls, type)
            and issubclass(cls, PreTrainedModel)
            and not getattr(cls, "_tt_post_init_patched", False)
        ):
            orig_init = cls.__init__

            def _patched_init(self, config, _orig=orig_init):
                _orig(self, config)
                if not hasattr(self, "all_tied_weights_keys"):
                    self.post_init()

            cls.__init__ = _patched_init

            orig_tie = cls.tie_weights

            def _patched_tie_weights(self, _orig=orig_tie, **kwargs):
                _orig(self)

            cls.tie_weights = _patched_tie_weights
            cls._tt_post_init_patched = True
            return


from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available PhoGPT-4B-Chat model variants for causal language modeling."""

    PHOGPT_4B_CHAT = "PhoGPT-4B-Chat"


class ModelLoader(ForgeModel):
    """VinAI PhoGPT-4B-Chat model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PHOGPT_4B_CHAT: ModelConfig(
            pretrained_model_name="vinai/PhoGPT-4B-Chat",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PHOGPT_4B_CHAT

    PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"
    sample_text = "Viết bài văn nghị luận xã hội về an toàn giao thông"

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
            model="PhoGPT-4B-Chat",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.n_layers = self.num_layers
            model_kwargs["config"] = config

        _patch_mpt_post_init()
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        prompt = self.PROMPT_TEMPLATE.format(instruction=self.sample_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
