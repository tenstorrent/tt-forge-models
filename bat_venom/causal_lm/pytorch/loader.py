# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BatVenom GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_gguf_pytorch_utils as _gguf_utils

import transformers.utils.import_utils as _tx_import_utils

_tx_import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
    importlib.metadata.packages_distributions()
)


def _find_real_load_gguf_checkpoint(fn):
    """Traverse patch chain to find the original transformers load_gguf_checkpoint."""
    seen = set()
    current = fn
    while True:
        fn_id = id(current)
        if fn_id in seen or not callable(current) or not hasattr(current, "__code__"):
            return current
        seen.add(fn_id)
        if (
            getattr(current, "__module__", "")
            == "transformers.modeling_gguf_pytorch_utils"
        ):
            return current
        freevars = current.__code__.co_freevars
        cells = current.__closure__ or ()
        next_fn = None
        for i, varname in enumerate(freevars):
            if i >= len(cells):
                break
            if (
                "load_gguf_checkpoint" in varname
                or "orig_load" in varname
                or "real_fn" in varname
                or "chain_fn" in varname
            ):
                try:
                    v = cells[i].cell_contents
                    if callable(v) and id(v) not in seen:
                        next_fn = v
                        break
                except ValueError:
                    pass
        if next_fn is None:
            globs = getattr(current, "__globals__", {})
            for varname in (
                "_orig_load_gguf_checkpoint",
                "_real_load_gguf_checkpoint",
                "_chain_fn",
                "_real_fn",
            ):
                v = globs.get(varname)
                if v is not None and callable(v) and id(v) not in seen:
                    next_fn = v
                    break
        if next_fn is None:
            return current
        current = next_fn


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
    """Available BatVenom model variants."""

    BAT_VENOM = "BatVenom"
    BAT_VENOM_V7 = "BatVenom-V7"


class ModelLoader(ForgeModel):
    """BatVenom GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BAT_VENOM: LLMModelConfig(
            pretrained_model_name="BrainDelay/BatVenom",
            max_length=128,
        ),
        ModelVariant.BAT_VENOM_V7: LLMModelConfig(
            pretrained_model_name="BrainDelay/BatVenom-V7",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BAT_VENOM

    GGUF_FILES = {
        ModelVariant.BAT_VENOM: "Mistral-BatVenom_V9.1_Q4_K_M.gguf",
        ModelVariant.BAT_VENOM_V7: "Mistral-BatVenom_V7.2_Q4_K_M.gguf",
    }

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self.GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BatVenom",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

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

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        _saved_fn = _gguf_utils.load_gguf_checkpoint
        _real_fn = _find_real_load_gguf_checkpoint(_saved_fn)

        def _patched(
            gguf_checkpoint_path, return_tensors=False, model_to_load=None, **kw
        ):
            return _real_fn(
                gguf_checkpoint_path,
                return_tensors=return_tensors,
                model_to_load=model_to_load,
            )

        _gguf_utils.load_gguf_checkpoint = _patched
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _saved_fn

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        self._fix_gguf_version_detection()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
