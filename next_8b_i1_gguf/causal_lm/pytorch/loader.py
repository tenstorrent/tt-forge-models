# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Next 8B i1 GGUF model loader implementation for causal language modeling.
"""
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


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
    """Available Next 8B i1 GGUF model variants for causal language modeling."""

    NEXT_8B_I1_Q4_K_M_GGUF = "NEXT_8B_I1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Next 8B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEXT_8B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/next-8b-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEXT_8B_I1_Q4_K_M_GGUF

    GGUF_FILE = "next-8b.i1-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="Next 8B i1 GGUF",
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
