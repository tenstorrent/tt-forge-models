# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternLM2 chat causal LM model loader implementation.
"""

import os
import shutil
import tempfile
from typing import Optional

import torch

# transformers is imported inside the methods so it binds to the pinned 4.46.3
# after RequirementsManager swaps it in, not the repo default at collection time.

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available InternLM2 model variants for causal language modeling."""

    INTERNLM2_CHAT_20B = "internlm2-chat-20b"


class ModelLoader(ForgeModel):
    """InternLM2 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.INTERNLM2_CHAT_20B: LLMModelConfig(
            pretrained_model_name="internlm/internlm2-chat-20b",
            max_length=256,
        )
    }

    DEFAULT_VARIANT = ModelVariant.INTERNLM2_CHAT_20B

    sample_text = "Who are you?"

    # Tokenizer files staged into a local copy. Do not snapshot the weight
    # shards -- a full internlm2-chat-20b download is tens of GB.
    _TOKENIZER_FILES = (
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenization_internlm2.py",
        "tokenization_internlm2_fast.py",
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None
        self._tokenizer_tmpdir = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InternLM2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _sanitize_spm_null_piece(self, spm_path: str) -> None:
        """Rewrite a null-char SentencePiece piece in ``spm_path`` in place.

        The checkpoint's SentencePiece model contains a single NORMAL piece whose
        surface is a literal null character. sentencepiece>=0.2.0 rejects it with
        ``RuntimeError: piece must not include null character``; transformers
        4.46.3 catches that inside ``_from_pretrained`` and silently returns
        ``False`` from ``AutoTokenizer.from_pretrained`` -- so the loader ends up
        with ``self.tokenizer = False`` and later hits
        ``'bool' object has no attribute 'apply_chat_template'``.

        Pinning sentencepiece<0.2 is not viable (no cp312 wheel; the sdist build
        fails), so instead we rewrite that one piece to a unique non-null
        placeholder. The token id is preserved, and the null piece never appears
        in real text, so tokenization/logits are unchanged. The rewrite is
        idempotent (the null byte is gone after the first pass).

        ``spm_path`` must be a caller-owned local copy, never an HF cache blob.
        """
        with open(spm_path, "rb") as f:
            blob = f.read()

        # Wire-format of a 1-byte NUL piece string (field 1, len 1, value 0x00).
        # If it's absent the file was already sanitized (or never affected).
        if b"\x0a\x01\x00" not in blob:
            return

        try:
            from sentencepiece import sentencepiece_model_pb2 as spb
        except Exception:
            from transformers.utils import sentencepiece_model_pb2 as spb

        model = spb.ModelProto()
        model.ParseFromString(blob)
        existing = {p.piece for p in model.pieces}
        changed = False
        for piece in model.pieces:
            if "\x00" in piece.piece:
                placeholder = "<0x00_nul>"
                while placeholder in existing:
                    placeholder += "_"
                existing.add(placeholder)
                piece.piece = placeholder
                changed = True
        if changed:
            with open(spm_path, "wb") as f:
                f.write(model.SerializeToString())

    def _prepare_tokenizer_dir(self) -> str:
        """Stage tokenizer files in a writable local dir; leave the HF cache alone.

        ``snapshot_download`` returns the shared cache snapshot, whose files are
        typically symlinks into ``blobs/``. Copying via ``os.path.realpath``
        dereferences those links so the cache is never written, including when
        the cache is read-only or shared over NFS. The temp dir is kept on
        ``self`` so ``vocab_file`` stays valid for the tokenizer's lifetime.
        """
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            self._variant_config.pretrained_model_name,
            allow_patterns=list(self._TOKENIZER_FILES),
        )

        self._tokenizer_tmpdir = tempfile.TemporaryDirectory(
            prefix="internlm2-tokenizer-"
        )
        tmp = self._tokenizer_tmpdir.name
        for name in self._TOKENIZER_FILES:
            src = os.path.join(snapshot, name)
            shutil.copy2(os.path.realpath(src), os.path.join(tmp, name))

        self._sanitize_spm_null_piece(os.path.join(tmp, "tokenizer.model"))
        return tmp

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        from transformers import AutoTokenizer

        tokenizer_dir = self._prepare_tokenizer_dir()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True, use_fast=False
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the InternLM2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The InternLM2 model for causal language modeling.
        """
        from transformers import AutoModelForCausalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the InternLM2 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        conversation = [{"role": "user", "content": self.sample_text}]
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads "
                f"across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.feed_forward.w1.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.w3.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.w2.weight] = ("batch", "model")

            shard_specs[layer.attention.wqkv.weight] = ("model", "batch")
            shard_specs[layer.attention.wo.weight] = ("batch", "model")
        shard_specs[model.output.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
