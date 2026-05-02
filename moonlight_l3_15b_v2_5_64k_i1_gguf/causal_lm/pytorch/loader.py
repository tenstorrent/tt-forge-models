# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moonlight-L3-15B-v2.5-64k i1 GGUF model loader implementation for causal language modeling.
"""
import inspect
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _restore_load_gguf_checkpoint():
    """Re-install load_gguf_checkpoint with model_to_load support if a broken patch is active.

    Multiple loaders patch load_gguf_checkpoint in a chain, each storing the
    previous function as _orig_load_gguf_checkpoint in their module globals.
    Walk the chain via a BFS over __globals__ entries whose names suggest they
    are GGUF-related, until we find the transformers original that accepts
    model_to_load.
    """
    current = _gguf_utils.load_gguf_checkpoint
    try:
        if "model_to_load" in inspect.signature(current).parameters:
            return
    except (ValueError, TypeError):
        pass

    # BFS: for each function, look in its __globals__ for gguf/orig-related callables.
    visited: set = set()
    queue = [current]
    real_fn = None
    while queue and real_fn is None:
        fn = queue.pop(0)
        fid = id(fn)
        if fid in visited:
            continue
        visited.add(fid)
        # Check if this function is the real original
        try:
            if "model_to_load" in inspect.signature(fn).parameters:
                if getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils":
                    real_fn = fn
                    break
                elif real_fn is None:
                    real_fn = fn
        except (ValueError, TypeError):
            pass
        # Enqueue candidates from this function's globals (names suggesting gguf/orig chain)
        for name, val in list((getattr(fn, "__globals__", None) or {}).items()):
            if not callable(val) or id(val) in visited:
                continue
            low = name.lower()
            if "load_gguf" in low or "gguf_check" in low or low.startswith("_orig"):
                queue.append(val)

    if real_fn is None:
        raise RuntimeError(
            "Cannot find load_gguf_checkpoint accepting model_to_load; "
            "update broken GGUF loaders for transformers 5.x compatibility."
        )

    def _patched(gguf_path, return_tensors=False, model_to_load=None):
        return real_fn(
            gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
        )

    _gguf_utils.load_gguf_checkpoint = _patched
    _config_utils.load_gguf_checkpoint = _patched
    _auto_tokenizer.load_gguf_checkpoint = _patched
    _tok_utils.load_gguf_checkpoint = _patched


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
    """Available Moonlight-L3-15B-v2.5-64k i1 GGUF model variants for causal language modeling."""

    MOONLIGHT_L3_15B_V2_5_64K_I1_Q4_K_M_GGUF = (
        "MOONLIGHT_L3_15B_V2_5_64K_I1_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Moonlight-L3-15B-v2.5-64k i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MOONLIGHT_L3_15B_V2_5_64K_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Moonlight-L3-15B-v2.5-64k-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOONLIGHT_L3_15B_V2_5_64K_I1_Q4_K_M_GGUF

    GGUF_FILE = "Moonlight-L3-15B-v2.5-64k.i1-Q4_K_M.gguf"

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
            model="Moonlight-L3-15B-v2.5-64k i1 GGUF",
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
        _restore_load_gguf_checkpoint()
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
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text
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
