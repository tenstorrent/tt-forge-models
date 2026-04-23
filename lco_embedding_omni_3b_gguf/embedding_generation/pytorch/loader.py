# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LCO-Embedding-Omni-3B GGUF model loader implementation for embedding generation.
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
    """Available LCO-Embedding-Omni-3B GGUF model variants for embedding generation."""

    LCO_EMBEDDING_OMNI_3B_Q4_K_M = "LCO_Embedding_Omni_3B_Q4_K_M"


class ModelLoader(ForgeModel):
    """LCO-Embedding-Omni-3B GGUF model loader implementation for embedding generation."""

    _VARIANTS = {
        ModelVariant.LCO_EMBEDDING_OMNI_3B_Q4_K_M: ModelConfig(
            pretrained_model_name="marksverdhei/LCO-Embedding-Omni-3B-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LCO_EMBEDDING_OMNI_3B_Q4_K_M

    GGUF_FILE = "LCO-Embedding-Omni-3B-Q4_K_M.gguf"

    sample_sentences = [
        "Scaling language-centric omnimodal representation learning.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LCO-Embedding-Omni-3B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        # transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time; if gguf
        # was installed after transformers was imported (e.g. via RequirementsManager),
        # it won't be in the mapping and the fallback reads gguf.__version__ which
        # gguf doesn't define, returning "N/A" and causing InvalidVersion.
        # Inject gguf into the mapping so importlib.metadata.version("gguf") is used.
        try:
            import transformers.utils.import_utils as _tu

            if "gguf" not in _tu.PACKAGE_DISTRIBUTION_MAPPING:
                _tu.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
        except Exception:
            pass

        # The GGUF file uses general.architecture=qwen2vl but contains only text
        # tensors (no vision tensors) — structurally identical to qwen2. Add qwen2vl
        # as an alias for qwen2 in the transformers GGUF loader mappings.
        try:
            import transformers.modeling_gguf_pytorch_utils as _gguf_utils

            for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
                if "qwen2" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
                    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                        "qwen2vl",
                        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen2"],
                    )
            if "qwen2vl" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
                _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
        except Exception:
            pass

        tokenizer_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs).eval()

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
