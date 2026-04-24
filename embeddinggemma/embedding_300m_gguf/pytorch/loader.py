# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EmbeddingGemma 300M GGUF model loader implementation for sentence embedding generation.
"""
import importlib.metadata

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
    """Available EmbeddingGemma 300M GGUF model variants for embedding generation."""

    EMBEDDINGGEMMA_300M_GGUF_Q4_K_M = "embeddinggemma-300m-gguf-Q4_K_M"


class ModelLoader(ForgeModel):
    """EmbeddingGemma 300M GGUF model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.EMBEDDINGGEMMA_300M_GGUF_Q4_K_M: ModelConfig(
            pretrained_model_name="second-state/embeddinggemma-300m-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EMBEDDINGGEMMA_300M_GGUF_Q4_K_M

    GGUF_FILE = "embeddinggemma-300m-Q4_K_M.gguf"

    sample_sentences = ["This is an example sentence for embedding generation"]

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager.

        transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
        is installed later, the mapping is stale and version detection falls back to
        gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
        """
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    @staticmethod
    def _patch_gemma_embedding_support():
        """Register gemma-embedding as an alias for gemma3_text in GGUF architecture mappings.

        The embeddinggemma-300m GGUF declares architecture 'gemma-embedding' which is
        not in transformers' GGUF_SUPPORTED_ARCHITECTURES. The underlying model is
        Gemma3-based (model_type=gemma3_text), so we alias it and fix the model_type
        returned by load_gguf_checkpoint so AutoConfig recognises the checkpoint.
        """
        import transformers.configuration_utils as _config_utils
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils
        import transformers.models.auto.tokenization_auto as _auto_tokenizer
        import transformers.tokenization_utils_tokenizers as _tok_utils
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
        from transformers.modeling_gguf_pytorch_utils import (
            load_gguf_checkpoint as _orig_load_gguf_checkpoint,
        )

        arch = "gemma-embedding"
        if arch not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
            _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append(arch)
        for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
            if "gemma3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                    arch,
                    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["gemma3"],
                )
        if "gemma3_text" in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS.setdefault(
                arch, GGUF_TO_FAST_CONVERTERS["gemma3_text"]
            )

        def _patched_load_gguf_checkpoint(*args, **kwargs):
            result = _orig_load_gguf_checkpoint(
                gguf_path, return_tensors=return_tensors
            )
            if result.get("config", {}).get("model_type") == arch:
                result["config"]["model_type"] = "gemma3_text"
            return result

        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="EmbeddingGemma-300M-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
        self._patch_gemma_embedding_support()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()
        self._patch_gemma_embedding_support()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
