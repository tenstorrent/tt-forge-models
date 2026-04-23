# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GENERator-eukaryote model loader implementation for causal language modeling.
"""
import importlib
import os
import pathlib
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
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


class ModelVariant(StrEnum):
    """Available GENERator-eukaryote model variants for causal language modeling."""

    GENERATOR_EUKARYOTE_1_2B_BASE = "GenerTeam/GENERator-eukaryote-1.2b-base"
    GENERATOR_V2_EUKARYOTE_1_2B_BASE = "GenerTeam/GENERator-v2-eukaryote-1.2b-base"


class ModelLoader(ForgeModel):
    """GENERator-eukaryote model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GENERATOR_EUKARYOTE_1_2B_BASE: LLMModelConfig(
            pretrained_model_name="GenerTeam/GENERator-eukaryote-1.2b-base",
            max_length=128,
        ),
        ModelVariant.GENERATOR_V2_EUKARYOTE_1_2B_BASE: LLMModelConfig(
            pretrained_model_name="GenerTeam/GENERator-v2-eukaryote-1.2b-base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERATOR_EUKARYOTE_1_2B_BASE

    # Sample genomic DNA sequence (length must be a multiple of 6 for the 6-mer tokenizer)
    sample_text = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GENERator-eukaryote",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_dnakmer_tokenizer_cache(model_name: str) -> None:
        # DNAKmerTokenizer sets bos/eos tokens before super().__init__(), which
        # breaks transformers >= 5.x where _special_tokens_map is an instance
        # variable initialized by super().__init__(). Fix the cached tokenizer
        # file and purge the stale import so the next load uses the fixed code.
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        org = model_name.split("/")[0]
        modules_base = pathlib.Path(hf_home) / "modules" / "transformers_modules" / org

        broken = (
            '        self.bos_token = "<s>"\n'
            '        self.eos_token = "</s>"\n'
            "        self.bos_token_id = self._convert_token_to_id(self.bos_token)\n"
            "        self.eos_token_id = self._convert_token_to_id(self.eos_token)\n"
            "        super().__init__(**kwargs)"
        )
        fixed = '        super().__init__(bos_token="<s>", eos_token="</s>", **kwargs)'

        for tokenizer_path in modules_base.rglob("tokenizer.py"):
            content = tokenizer_path.read_text()
            if "DNAKmerTokenizer" in content and broken in content:
                tokenizer_path.write_text(content.replace(broken, fixed))
                for key in list(sys.modules.keys()):
                    if "transformers_modules" in key:
                        del sys.modules[key]
                importlib.invalidate_caches()

    def _load_tokenizer(self, dtype_override=None):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
                padding_side="left",
            )
        except AttributeError as e:
            if "_special_tokens_map" not in str(e):
                raise
            self._patch_dnakmer_tokenizer_cache(
                self._variant_config.pretrained_model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
                padding_side="left",
            )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        return inputs
