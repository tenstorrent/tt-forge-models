# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gecko text embedding model loader implementation.

Gecko is a compact text embedding model from Google (110M parameters),
distributed as TFLite via litert-community/Gecko-110m-en on HuggingFace.
The underlying architecture is a T5-like encoder (12 layers, 768 hidden,
2048 FFN, 12 heads, gated-gelu FFN). This loader creates a T5EncoderModel
with matching config for compilation through the XLA/torch.compile path.
"""
import torch
from transformers import T5EncoderModel, T5Config
from typing import Optional
from huggingface_hub import hf_hub_download

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
    """Available Gecko model variants for embedding generation."""

    GECKO_256 = "gecko-256"


class ModelLoader(ForgeModel):
    """Gecko text embedding model loader implementation."""

    _VARIANTS = {
        ModelVariant.GECKO_256: ModelConfig(
            pretrained_model_name="litert-community/Gecko-110m-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GECKO_256

    sample_sentences = ["This is an example sentence for embedding generation"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Gecko",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return a T5EncoderModel matching the Gecko architecture."""
        config = T5Config(
            vocab_size=32128,
            d_model=768,
            d_kv=64,
            d_ff=2048,
            num_heads=12,
            num_layers=12,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=256,
            feed_forward_proj="gated-gelu",
            is_encoder_decoder=False,
            use_cache=False,
        )

        model = T5EncoderModel(config)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def _load_tokenizer(self):
        import sentencepiece as spm

        repo_id = self._variant_config.pretrained_model_name
        sp_model_path = hf_hub_download(repo_id=repo_id, filename="sentencepiece.model")
        self.tokenizer = spm.SentencePieceProcessor(model_file=sp_model_path)
        return self.tokenizer

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Gecko model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        encoded = self.tokenizer.encode(self.sample_sentences[0], out_type=int)

        max_length = 256
        if len(encoded) > max_length:
            encoded = encoded[:max_length]

        input_ids = encoded + [0] * (max_length - len(encoded))
        attention_mask = [1] * len(encoded) + [0] * (max_length - len(encoded))

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
