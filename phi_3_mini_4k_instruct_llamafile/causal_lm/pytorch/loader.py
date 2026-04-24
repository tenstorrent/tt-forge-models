# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mozilla-AI Phi-3-mini-4k-instruct-llamafile model loader implementation for causal language modeling.
"""
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
import transformers.utils.import_utils as _import_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)

# transformers 5.2+ omits 'gguf' from PACKAGE_DISTRIBUTION_MAPPING, causing
# is_gguf_available() to crash with InvalidVersion when the gguf package is
# installed but lacks a __version__ attribute.
if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
    _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ("gguf",)

_GGUF_MAGIC = b"GGUF"


def _extract_gguf_if_llamafile(path: str) -> str:
    """Return path to a raw GGUF file, extracting from a llamafile zip if needed."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == _GGUF_MAGIC:
        return path
    gguf_path = Path(path).parent / (Path(path).name + ".gguf")
    if not gguf_path.exists():
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.filename.endswith(".gguf"):
                    with zf.open(info) as src, open(gguf_path, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
                    break
            else:
                raise ValueError(f"No .gguf file found inside {path}")
    return str(gguf_path)


def _patched_load_gguf_checkpoint(*args, **kwargs):
    gguf_path = _extract_gguf_if_llamafile(gguf_path)
    return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)


_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
    """Available Phi-3-mini-4k-instruct-llamafile model variants for causal language modeling."""

    PHI_3_MINI_4K_INSTRUCT_Q4_K_M_LLAMAFILE = "Phi_3_mini_4k_instruct_Q4_K_M_llamafile"


class ModelLoader(ForgeModel):
    """Mozilla-AI Phi-3-mini-4k-instruct-llamafile model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PHI_3_MINI_4K_INSTRUCT_Q4_K_M_LLAMAFILE: LLMModelConfig(
            pretrained_model_name="mozilla-ai/Phi-3-mini-4k-instruct-llamafile",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PHI_3_MINI_4K_INSTRUCT_Q4_K_M_LLAMAFILE

    GGUF_FILE = "Phi-3-mini-4k-instruct.Q4_K_M.llamafile"

    sample_text = (
        "Can you provide ways to eat combinations of bananas and dragonfruits?"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Phi-3-mini-4k-instruct-llamafile",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
