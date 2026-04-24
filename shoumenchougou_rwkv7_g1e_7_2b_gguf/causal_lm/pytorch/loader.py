# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
shoumenchougou RWKV7 G1e 7.2B GGUF model loader implementation for causal language modeling.

Transformers 5.x does not support loading rwkv7 from GGUF checkpoints, and
the fla library (which implements RWKV7) requires Triton unavailable on
Tenstorrent hardware.  We therefore:
  1. Use a custom pure-PyTorch RWKV7 model (model.py in this package) that
     reads and dequantises the GGUF weights directly.
  2. Load the tokenizer using the rwkv package's TRIE_TOKENIZER with the
     bundled RWKV World v2 vocabulary file, bypassing transformers' GGUF
     tokenizer conversion path which doesn't support RWKV.
"""
import os
import torch
from huggingface_hub import hf_hub_download
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


class _RWKVWorldTokenizer:
    """Minimal tokenizer wrapper around RWKV World v2 TRIE_TOKENIZER."""

    def __init__(self, trie_tok):
        self._tok = trie_tok
        self.pad_token = "\x00"
        self.eos_token = "\x00"
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(
        self, texts, return_tensors=None, padding=True, truncation=True, max_length=None
    ):
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self._tok.encode(t) for t in texts]

        if truncation and max_length is not None:
            encoded = [e[:max_length] for e in encoded]

        max_len = max(len(e) for e in encoded)

        padded = []
        masks = []
        for e in encoded:
            pad_len = max_len - len(e)
            padded.append(e + [self.pad_token_id] * pad_len)
            masks.append([1] * len(e) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class ModelVariant(StrEnum):
    """Available shoumenchougou RWKV7 G1e 7.2B GGUF model variants for causal language modeling."""

    RWKV7_G1E_7_2B_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """shoumenchougou RWKV7 G1e 7.2B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.RWKV7_G1E_7_2B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="shoumenchougou/RWKV7-G1e-7.2B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RWKV7_G1E_7_2B_Q4_K_M

    GGUF_FILE = "rwkv7-g1e-7.2b-Q4_K_M-better_quantization-20260319.gguf"

    sample_text = "Once upon a time, in a land far away,"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers
        self._gguf_path = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="shoumenchougou RWKV7 G1e 7.2B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_gguf_path(self) -> str:
        if self._gguf_path is None:
            self._gguf_path = hf_hub_download(
                self._variant_config.pretrained_model_name, self.GGUF_FILE
            )
        return self._gguf_path

    def _load_tokenizer(self, dtype_override=None):
        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
        import rwkv.rwkv_tokenizer as _rwkv_tok_mod

        vocab_file = os.path.join(
            os.path.dirname(os.path.abspath(_rwkv_tok_mod.__file__)),
            "rwkv_vocab_v20230424.txt",
        )
        self.tokenizer = _RWKVWorldTokenizer(TRIE_TOKENIZER(vocab_file))
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from .model import load_rwkv7_from_gguf

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = self._get_gguf_path()

        model = load_rwkv7_from_gguf(gguf_path)

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = model.cfg
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
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
        from .model import load_rwkv7_from_gguf

        gguf_path = self._get_gguf_path()
        model = load_rwkv7_from_gguf(gguf_path)
        self.config = model.cfg
        return self.config
