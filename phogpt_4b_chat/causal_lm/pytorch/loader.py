# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VinAI PhoGPT-4B-Chat causal language model loader implementation.
"""

import sys
import types
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import HF_MODULES_CACHE

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available PhoGPT-4B-Chat model variants for causal language modeling."""

    PHOGPT_4B_CHAT = "PhoGPT-4B-Chat"


class ModelLoader(ForgeModel):
    """VinAI PhoGPT-4B-Chat model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PHOGPT_4B_CHAT: ModelConfig(
            pretrained_model_name="vinai/PhoGPT-4B-Chat",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PHOGPT_4B_CHAT

    PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"
    sample_text = "Viết bài văn nghị luận xã hội về an toàn giao thông"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PhoGPT-4B-Chat",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _ensure_flash_attn_triton_stub():
        # transformers' file scanner fails on `from .flash_attn_triton import ...` if the
        # stub file doesn't exist, even inside a try/except. Create a placeholder so the
        # scanner can resolve the relative import without error.
        stub_content = (
            "# Stub: flash_attn_triton not available; model uses attn_impl='torch'\n"
            "def flash_attn_func(*args, **kwargs):\n"
            "    raise RuntimeError(\"flash_attn_triton unavailable; use attn_impl='torch'\")\n"
        )
        modules_root = Path(HF_MODULES_CACHE) / "transformers_modules"
        for attention_py in modules_root.glob("vinai/PhoGPT*/**/attention.py"):
            stub = attention_py.parent / "flash_attn_triton.py"
            if not stub.exists():
                stub.write_text(stub_content)

        # attention.py checks `flash_attn.__version__` outside its try/except, so a
        # namespace package (no __version__) causes AttributeError at module load time.
        # Ensure flash_attn is either fully installed or replaced with a versioned stub.
        try:
            import flash_attn as _fa  # noqa: F401

            if not hasattr(_fa, "__version__"):
                _fa.__version__ = "0.0.0"
        except ImportError:
            fake = types.ModuleType("flash_attn")
            fake.__version__ = "0.0.0"
            sys.modules["flash_attn"] = fake

    @staticmethod
    def _patch_transformers_compat():
        # transformers 5.x removed the specialized Llama rotary embedding subclasses.
        # modeling_mpt.py imports them at module level even when alibi (not rope) is used,
        # so provide aliases pointing to the unified LlamaRotaryEmbedding.
        import transformers.models.llama.modeling_llama as llama_module

        for cls_name in (
            "LlamaLinearScalingRotaryEmbedding",
            "LlamaDynamicNTKScalingRotaryEmbedding",
        ):
            if not hasattr(llama_module, cls_name):
                setattr(llama_module, cls_name, llama_module.LlamaRotaryEmbedding)

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self._ensure_flash_attn_triton_stub()
        self._patch_transformers_compat()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        config.attn_config["attn_impl"] = "torch"
        if self.num_layers is not None:
            config.n_layers = self.num_layers

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        prompt = self.PROMPT_TEMPLATE.format(instruction=self.sample_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self._ensure_flash_attn_triton_stub()
        self._patch_transformers_compat()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        self.config.attn_config["attn_impl"] = "torch"
        return self.config
