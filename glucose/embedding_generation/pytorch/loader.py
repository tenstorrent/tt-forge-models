# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLuCoSE-base-ja-v2 model loader implementation for sentence embedding generation.
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
    """Available GLuCoSE model variants for embedding generation."""

    GLUCOSE_BASE_JA_V2 = "glucose-base-ja-v2"


def _patch_mluke_tokenizer():
    """Fix transformers 5.x MLukeTokenizer bug.

    In transformers 5.x, MLukeTokenizer was refactored to use TokenizersBackend.
    When loading a BPE SentencePiece model, SentencePieceExtractor.extract() returns
    vocab as a {token: spm_id} dict.  The original __init__ always passes this to
    Unigram() which requires List[Tuple[str, float]], causing TypeError.

    Fix: detect the BPE dict case, remap IDs to fairseq alignment, build a proper
    BPE tokenizer, then replace the dummy Unigram created by the original __init__.
    """
    import transformers.models.mluke.tokenization_mluke as _mod
    from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, processors
    from tokenizers.models import BPE

    if getattr(_mod.MLukeTokenizer, "_bpe_fix_applied", False):
        return

    _orig_init = _mod.MLukeTokenizer.__init__

    def _patched_init(self, *args, vocab=None, **kwargs):
        merges = kwargs.pop("merges", None)
        unk_id = kwargs.pop("unk_id", 0)

        if not (isinstance(vocab, dict) and merges is not None):
            # Unigram or no-vocab path: use original code unchanged
            return _orig_init(self, *args, vocab=vocab, **kwargs)

        # BPE dict vocab from SentencePieceExtractor.
        # SPM ordering: <unk>=0, <s>=1, </s>=2, real_tokens=3+
        # fairseq ordering: <s>=0, <pad>=1, </s>=2, <unk>=3, real_tokens=4+
        # (fairseq_offset=1: fairseq_id = spm_id + 1 for real tokens)
        _fairseq_special = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        fairseq_vocab = {}
        for token, spm_id in vocab.items():
            fairseq_vocab[token] = _fairseq_special.get(token, spm_id + 1)
        # <pad> is not in the SPM vocab; add it at fairseq ID 1
        fairseq_vocab.setdefault("<pad>", 1)

        _bpe = Tokenizer(BPE(vocab=fairseq_vocab, merges=merges, unk_token="<unk>"))
        # Include NFKC so fullwidth characters (e.g. '？') are normalised the same
        # way the underlying SentencePiece model normalises them before tokenising.
        _bpe.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.Replace("``", '"'),
                normalizers.Replace("''", '"'),
            ]
        )
        _bpe.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="always"
        )
        _bpe.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

        # Call original __init__ with vocab=None so it runs all other initialisation
        # (entity_vocab, fairseq_tokens_to_ids, super().__init__, _post_init) without
        # crashing on Unigram(dict).  This creates a 1-token dummy Unigram internally.
        _orig_init(self, *args, vocab=None, **kwargs)

        # Replace the dummy Unigram tokenizer with the proper BPE tokenizer.
        self._tokenizer = _bpe

        # Recompute vocab metadata from BPE tokenizer.
        self._vocab_size = self._tokenizer.get_vocab_size(with_added_tokens=False)
        self.fairseq_tokens_to_ids["<mask>"] = (
            self._vocab_size + self.fairseq_offset
        )
        self.fairseq_ids_to_tokens = {
            v: k for k, v in self.fairseq_tokens_to_ids.items()
        }

        # Re-apply post_processor using correct IDs from the BPE tokenizer.
        cls_id = self._tokenizer.token_to_id(self.cls_token)  # 0
        sep_id = self._tokenizer.token_to_id(self.sep_token)  # 2
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.cls_token}:0 $A:0 {self.sep_token}:0",
            pair=(
                f"{self.cls_token}:0 $A:0 {self.sep_token}:0 "
                f"{self.sep_token}:0 $B:1 {self.sep_token}:1"
            ),
            special_tokens=[
                (self.cls_token, cls_id),
                (self.sep_token, sep_id),
            ],
        )

    _mod.MLukeTokenizer.__init__ = _patched_init
    _mod.MLukeTokenizer._bpe_fix_applied = True


class ModelLoader(ForgeModel):
    """GLuCoSE-base-ja-v2 model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.GLUCOSE_BASE_JA_V2: ModelConfig(
            pretrained_model_name="pkshatech/GLuCoSE-base-ja-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLUCOSE_BASE_JA_V2

    sample_sentences = ["query: PKSHAはどんな会社ですか？"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GLuCoSE-base-ja-v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_mluke_tokenizer()

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
