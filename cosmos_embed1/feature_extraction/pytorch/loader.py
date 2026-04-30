# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Embed1 model loader implementation for video-text feature extraction.
"""

import json
import os

import torch
from safetensors.torch import load_file as _load_safetensors
from transformers import AutoModel, AutoTokenizer
from transformers.utils import cached_file as _hf_cached_file
from typing import Optional

from .src import (
    model_utils as _model_utils,
)  # noqa: F401  # patches init_weights for transformers 5.x

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
    """Available Cosmos Embed1 model variants."""

    COSMOS_EMBED1_448P = "Cosmos-Embed1-448p"


class ModelLoader(ForgeModel):
    """Cosmos Embed1 model loader for video-text feature extraction."""

    _VARIANTS = {
        ModelVariant.COSMOS_EMBED1_448P: ModelConfig(
            pretrained_model_name="nvidia/Cosmos-Embed1-448p",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COSMOS_EMBED1_448P

    sample_texts = [
        "A car driving on a highway",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Cosmos-Embed1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # ignore_mismatched_sizes=True: EvaViTG's pos_embed is interpolated from a
        # 224px checkpoint ([1,257,1408]) to 448px ([1,1025,1408]).  transformers 5.x
        # no longer calls _load_state_dict_pre_hooks during from_pretrained, so the
        # PositionalEmbeddingHook never runs and the size mismatch would raise.  We
        # suppress the error here and apply the interpolation manually below.
        model_kwargs = {"trust_remote_code": True, "ignore_mismatched_sizes": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.x initialises models on meta device; EvaViTG/VisionTransformer.__init__
        # calls torch.linspace(...).item() to build drop-path rate lists, which fails on meta
        # tensors.  Patch .item() to return 0.0 for meta scalars — the actual rates are not
        # stored as checkpoint parameters, and in eval mode stochastic depth is always a no-op.
        _orig_item = torch.Tensor.item

        def _meta_safe_item(t):
            return 0.0 if t.is_meta else _orig_item(t)

        torch.Tensor.item = _meta_safe_item
        try:
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        finally:
            torch.Tensor.item = _orig_item

        # Apply pos_embed interpolation via PyTorch's load_state_dict, which does
        # call _load_state_dict_pre_hooks (including PositionalEmbeddingHook).
        index_file = _hf_cached_file(pretrained_model_name, "model.safetensors.index.json")
        with open(index_file) as f:
            index = json.load(f)
        shard_name = index["weight_map"]["visual_encoder.pos_embed"]
        shard_path = os.path.join(os.path.dirname(index_file), shard_name)
        pos_embed_ckpt = _load_safetensors(shard_path, device="cpu")["visual_encoder.pos_embed"]
        model.visual_encoder.load_state_dict({"pos_embed": pos_embed_ckpt}, strict=False)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        text_inputs = self.tokenizer(
            self.sample_texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        batch_size = text_inputs["input_ids"].shape[0]
        video_dtype = dtype_override if dtype_override is not None else torch.float32
        videos = torch.randn(batch_size, 8, 3, 448, 448, dtype=video_dtype)

        inputs = {
            "videos": videos,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

        return inputs
