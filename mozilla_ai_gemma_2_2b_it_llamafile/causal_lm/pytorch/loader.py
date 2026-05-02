# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mozilla-AI gemma-2-2b-it-llamafile model loader implementation for causal language modeling.

The mozilla-ai/gemma-2-2b-it-llamafile HuggingFace repo contains only llamafile
executables (GGUF models wrapped in a shell script), not standard HuggingFace model
files. This loader uses the equivalent GGUF model from bartowski/gemma-2-2b-it-GGUF
which provides the same Gemma 2 2B IT weights in GGUF Q4_K_M quantization.
"""
import contextlib
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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


def _find_real_load_gguf_checkpoint():
    """Find the original load_gguf_checkpoint from transformers.

    Other loaders patch load_gguf_checkpoint at module import time with versions
    that don't accept the model_to_load kwarg added in transformers 5.x. Use gc to
    find the original function still referenced by those patchers' _orig_* variables.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    _REAL_MODULE = "transformers.modeling_gguf_pytorch_utils"
    _REAL_NAME = "load_gguf_checkpoint"
    for obj in gc.get_objects():
        try:
            if (
                callable(obj)
                and getattr(obj, "__name__", None) == _REAL_NAME
                and getattr(obj, "__module__", None) == _REAL_MODULE
            ):
                return obj
        except Exception:
            pass
    return None


@contextlib.contextmanager
def _use_real_load_gguf_checkpoint():
    """Temporarily restore the original load_gguf_checkpoint for the duration of this context."""
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.models.auto.tokenization_auto as _tok_auto
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_utils as _modeling_utils

    real_fn = _find_real_load_gguf_checkpoint()
    if real_fn is None:
        yield
        return

    saved = {}
    for mod in (_gguf_utils, _tok_auto, _config_utils, _modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            saved[mod] = mod.load_gguf_checkpoint
            mod.load_gguf_checkpoint = real_fn
    try:
        yield
    finally:
        for mod, fn in saved.items():
            mod.load_gguf_checkpoint = fn


class ModelVariant(StrEnum):
    """Available Mozilla-AI gemma-2-2b-it-llamafile model variants for causal language modeling."""

    GEMMA_2_2B_IT_LLAMAFILE = "gemma-2-2b-it-llamafile"


class ModelLoader(ForgeModel):
    """Mozilla-AI gemma-2-2b-it-llamafile model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_2_2B_IT_LLAMAFILE: LLMModelConfig(
            pretrained_model_name="bartowski/gemma-2-2b-it-GGUF",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_2_2B_IT_LLAMAFILE

    _GGUF_FILE = "gemma-2-2b-it-Q4_K_M.gguf"

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="gemma-2-2b-it-llamafile",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"gguf_file": self._GGUF_FILE}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        with _use_real_load_gguf_checkpoint():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                **tokenizer_kwargs,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": self._GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            with _use_real_load_gguf_checkpoint():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self._GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _use_real_load_gguf_checkpoint():
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                **model_kwargs,
            )

        model.eval()
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

        prompts = [text] * batch_size

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs
