# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LUKE model loader implementation for sequence classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


def _patch_mluke_tokenizer():
    """Fix transformers 5.x MLukeTokenizer BPE init crash.

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


class ModelVariant(StrEnum):
    """Available LUKE model variants for sequence classification."""

    MIZUIRO_SAKURA_LUKE_JAPANESE_LARGE_SENTIMENT_ANALYSIS_WRIME = (
        "mizuiro_sakura_luke_japanese_large_sentiment_analysis_wrime"
    )


class ModelLoader(ForgeModel):
    """LUKE model loader implementation for sequence classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MIZUIRO_SAKURA_LUKE_JAPANESE_LARGE_SENTIMENT_ANALYSIS_WRIME: LLMModelConfig(
            pretrained_model_name="Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
            max_length=512,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = (
        ModelVariant.MIZUIRO_SAKURA_LUKE_JAPANESE_LARGE_SENTIMENT_ANALYSIS_WRIME
    )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.max_length = 512
        self.tokenizer = None
        self.review = "今日はとても楽しい一日でした。"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="LUKE",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load LUKE model for sequence classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The model instance.
        """
        _patch_mluke_tokenizer()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for LUKE sequence classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            # Ensure tokenizer is initialized
            self.load_model(dtype_override=dtype_override)

        # Data preprocessing
        inputs = self.tokenizer(
            self.review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for sequence classification.

        Args:
            co_out: Model output
        """
        predicted_value = co_out[0].argmax(-1).item()

        print(f"Predicted Sentiment: {self.model.config.id2label[predicted_value]}")
