# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM 4.7 Flash Derestricted i1 GGUF model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _find_original_load_gguf_checkpoint():
    """Walk the monkey-patch chain to find the real transformers load_gguf_checkpoint.

    Broken patches in this repo typically store the prior function as a module-level
    global (e.g. ``_orig_load_gguf_checkpoint``) rather than a closure variable, so
    we inspect both ``__closure__`` and ``__globals__``.
    """
    import inspect
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    # Names commonly used to hold the prior function in broken patches.
    _ORIG_NAMES = (
        "_orig_load_gguf_checkpoint",
        "_orig",
        "orig_load",
        "_real",
        "_current_load",
        "_base_load",
    )

    fn = gguf_utils.load_gguf_checkpoint
    seen = set()
    while True:
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)

        try:
            src = inspect.getfile(fn)
        except (TypeError, OSError):
            break

        # If the source is from transformers itself (not a model loader), we found it.
        if "tt_forge_models" not in src and "worktrees" not in src:
            return fn

        # __wrapped__ convention (used by our own good patches).
        if hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
            continue

        # Closure variables (enclosing-scope pattern).
        unwrapped = None
        if fn.__closure__ and fn.__code__.co_freevars:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                if name.startswith(("_orig", "orig_", "_real", "_current", "_base")):
                    try:
                        val = cell.cell_contents
                        if callable(val) and id(val) not in seen:
                            unwrapped = val
                            break
                    except ValueError:
                        pass

        # Module-global pattern (broken patches use ``_orig_load_gguf_checkpoint``
        # defined at module scope, so it appears in ``fn.__globals__``).
        if unwrapped is None and hasattr(fn, "__globals__"):
            for name in _ORIG_NAMES:
                val = fn.__globals__.get(name)
                if callable(val) and id(val) not in seen:
                    unwrapped = val
                    break

        if unwrapped is not None:
            fn = unwrapped
            continue

        break

    return fn


def _patch_transformers_deepseek_gguf():
    """Monkey-patch transformers to add deepseek2/deepseek_v2 GGUF support.

    1. Registers deepseek2/deepseek_v2 tokenizer converters.
    2. Replaces the current (possibly broken) load_gguf_checkpoint with a
       wrapper that calls the original transformers implementation and accepts
       all kwargs including model_to_load added in transformers 5.x.
    3. Wraps get_gguf_hf_weights_map to remap the deepseek_v2 model_type
       (used by mradermacher GGUFs) to deepseek2 (known to gguf-py).

    Applied lazily before model/tokenizer loading so it always runs last.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.modeling_utils as modeling_utils
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    for key in ("deepseek2", "deepseek_v2"):
        if key not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS[key] = GGUFQwen2Converter

    _orig = _find_original_load_gguf_checkpoint()

    def _patched(*args, **kwargs):
        return _orig(*args, **kwargs)

    _patched.__wrapped__ = _orig
    gguf_utils.load_gguf_checkpoint = _patched
    if hasattr(modeling_utils, "load_gguf_checkpoint"):
        modeling_utils.load_gguf_checkpoint = _patched

    # Patch get_gguf_hf_weights_map to remap deepseek_v2 → deepseek2 so that
    # gguf-py MODEL_ARCH_NAMES lookup succeeds. Guard against double-patching.
    if not getattr(gguf_utils.get_gguf_hf_weights_map, "_deepseek_v2_patched", False):
        _orig_get_map = gguf_utils.get_gguf_hf_weights_map

        def _patched_get_map(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            # Resolve model_type early so we can remap before the original does.
            if model_type is None and hasattr(hf_model, "config"):
                model_type = hf_model.config.model_type
            if model_type == "deepseek_v2":
                model_type = "deepseek2"
            return _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

        _patched_get_map.__wrapped__ = _orig_get_map
        _patched_get_map._deepseek_v2_patched = True
        gguf_utils.get_gguf_hf_weights_map = _patched_get_map


class ModelVariant(StrEnum):
    """Available GLM 4.7 Flash Derestricted i1 GGUF model variants for causal language modeling."""

    GLM_4_7_FLASH_DERESTRICTED_I1_GGUF = "4_7_Flash_Derestricted_i1_GGUF"


class ModelLoader(ForgeModel):
    """GLM 4.7 Flash Derestricted i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_DERESTRICTED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/GLM-4.7-Flash-Derestricted-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_DERESTRICTED_I1_GGUF

    GGUF_FILE = "GLM-4.7-Flash-Derestricted.i1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="GLM 4.7 Flash Derestricted i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_transformers_deepseek_gguf()
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
        _patch_transformers_deepseek_gguf()
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

        messages = [{"role": "user", "content": self.sample_text}]
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
