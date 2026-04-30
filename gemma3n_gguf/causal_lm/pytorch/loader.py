# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 3n GGUF model loader implementation for causal language modeling.
"""
import contextlib
import torch
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


def _find_real_load_gguf_checkpoint(fn):
    """Walk a chain of monkey-patched wrappers to find the real transformers function.

    Many GGUF loaders patch load_gguf_checkpoint with fixed-arg wrappers that
    do not accept model_to_load. Walk through closure variables and module-global
    references until we reach the function whose __name__ is 'load_gguf_checkpoint'
    in the transformers module.
    """
    seen = set()
    while callable(fn):
        fn_id = id(fn)
        if fn_id in seen:
            return fn
        seen.add(fn_id)

        if (getattr(fn, '__name__', '') == 'load_gguf_checkpoint'
                and 'transformers' in getattr(fn, '__module__', '')):
            return fn

        moved = False
        # Closure pattern: good wrappers capture orig_load as a local variable
        if fn.__closure__:
            for cell in fn.__closure__:
                try:
                    v = cell.cell_contents
                    if callable(v) and id(v) not in seen:
                        fn = v
                        moved = True
                        break
                except ValueError:
                    pass

        # Module-global pattern: broken wrappers import the original as
        # _orig_load_gguf_checkpoint in their loader module's globals
        if not moved:
            globs = getattr(fn, '__globals__', {})
            orig = globs.get('_orig_load_gguf_checkpoint')
            if callable(orig) and id(orig) not in seen:
                fn = orig
                moved = True

        if not moved:
            return fn

    return fn


def _register_gemma3n_gguf_support():
    """Register gemma3n in the GGUF mappings once (idempotent).

    Transformers 5.x has Gemma3nForCausalLM but lacks GGUF loading support
    for the gemma3n architecture. Register it using the same field layout as
    gemma3, and alias the tokenizer converter to GGUFGemmaConverter.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        Gemma2TensorProcessor,
    )

    if "gemma3n" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("gemma3n")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["gemma3n"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.sliding_window": "sliding_window",
        "vocab_size": "vocab_size",
    }

    TENSOR_PROCESSORS["gemma3n"] = Gemma2TensorProcessor

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "gemma3_text" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gemma3n"] = GGUF_TO_FAST_CONVERTERS["gemma3_text"]
        GGUF_TO_FAST_CONVERTERS["gemma3n_text"] = GGUF_TO_FAST_CONVERTERS["gemma3_text"]


# Register gemma3n GGUF support at import time (idempotent)
_register_gemma3n_gguf_support()


@contextlib.contextmanager
def _gemma3n_gguf_load_context():
    """Context manager that temporarily installs a correct load_gguf_checkpoint.

    Many other GGUF loaders install fixed-arg patches that do not accept the
    model_to_load keyword argument required by transformers 5.x. By the time
    this model is loaded at test time, the module attribute may point to one of
    those broken patches. This context manager temporarily replaces the module
    attribute with a correct wrapper that:
      - Calls the real transformers function directly (found by chain-walking)
      - Remaps model_type "gemma3n" -> "gemma3n_text" so AutoModelForCausalLM
        instantiates Gemma3nForCausalLM instead of the multimodal class
      - Remaps model_type "gemma3n_text" -> "gemma3n" in get_gguf_hf_weights_map
        so gguf-py's MODEL_ARCH_NAMES lookup succeeds
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    # Find the real transformers function by walking past all broken wrappers
    true_orig = _find_real_load_gguf_checkpoint(gguf_utils.load_gguf_checkpoint)

    def _patched_load(*args, **kwargs):
        result = true_orig(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "gemma3n":
            result["config"]["model_type"] = "gemma3n_text"
        return result

    # Patch get_gguf_hf_weights_map so the real function's internal call works
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("gemma3n_text", "gemma3n"):
            model_type = "gemma3n"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    saved_load = gguf_utils.load_gguf_checkpoint
    gguf_utils.load_gguf_checkpoint = _patched_load
    gguf_utils.get_gguf_hf_weights_map = _patched_get_map
    try:
        yield
    finally:
        gguf_utils.load_gguf_checkpoint = saved_load
        gguf_utils.get_gguf_hf_weights_map = orig_get_map


class ModelVariant(StrEnum):
    """Available Gemma 3n GGUF model variants for causal language modeling."""

    GEMMA_3N_E4B_IT_Q4_K_M = "Gemma_3n_E4B_IT_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Gemma 3n GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3N_E4B_IT_Q4_K_M: LLMModelConfig(
            pretrained_model_name="NexaAI/gemma-3n",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3N_E4B_IT_Q4_K_M

    GGUF_FILE = "gemma-3n-E4B-it-Q4_K_M-full-vocab.gguf"

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
            model="Gemma 3n GGUF",
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

        with _gemma3n_gguf_load_context():
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
            with _gemma3n_gguf_load_context():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _gemma3n_gguf_load_context():
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

        input_prompt = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        input_text = self.tokenizer.apply_chat_template(
            input_prompt,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            [input_text],
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
        with _gemma3n_gguf_load_context():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
