# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek R1 Distill Qwen 1.5B Uncensored GGUF model loader implementation for causal language modeling.
"""
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

_GGUF_CACHE_DIR = "/tmp/hf_cache_deepseek_r1_distill_qwen_1_5b_uncensored"


def _find_real_load_gguf_checkpoint():
    fn = _gguf_utils.load_gguf_checkpoint
    visited = set()
    while id(fn) not in visited:
        visited.add(id(fn))
        source = getattr(getattr(fn, "__code__", None), "co_filename", "")
        if (
            "modeling_gguf_pytorch_utils.py" in source
            and "tt_forge_models" not in source
        ):
            return fn
        next_fn = None
        co_names = getattr(getattr(fn, "__code__", None), "co_names", ())
        fn_globals = getattr(fn, "__globals__", {})
        for name in co_names:
            if "orig" in name.lower() and name in fn_globals:
                val = fn_globals[name]
                if callable(val) and hasattr(val, "__code__"):
                    next_fn = val
                    break
        if next_fn is None and getattr(fn, "__closure__", None):
            free_vars = getattr(getattr(fn, "__code__", None), "co_freevars", ())
            for i, name in enumerate(free_vars):
                if "orig" in name.lower() and i < len(fn.__closure__):
                    try:
                        val = fn.__closure__[i].cell_contents
                        if callable(val):
                            next_fn = val
                            break
                    except ValueError:
                        pass
        if next_fn is None:
            break
        fn = next_fn
    return fn


def _ensure_gguf_checkpoint_accepts_model_to_load():
    real_fn = _find_real_load_gguf_checkpoint()
    for _mod in (_gguf_utils, _config_utils, _auto_tokenizer, _tok_utils):
        _mod.load_gguf_checkpoint = real_fn


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
    """Available DeepSeek R1 Distill Qwen 1.5B Uncensored GGUF model variants for causal language modeling."""

    DEEPSEEK_R1_DISTILL_QWEN_1_5B_UNCENSORED_Q4_K_M_GGUF = (
        "deepseek_r1_distill_qwen_1_5b_uncensored_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """DeepSeek R1 Distill Qwen 1.5B Uncensored GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_R1_DISTILL_QWEN_1_5B_UNCENSORED_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_R1_DISTILL_QWEN_1_5B_UNCENSORED_Q4_K_M_GGUF

    GGUF_FILE = "DeepSeek-R1-Distill-Qwen-1.5B-uncensored.Q4_K_M.gguf"

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
            model="DeepSeek R1 Distill Qwen 1.5B Uncensored GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _ensure_gguf_checkpoint_accepts_model_to_load()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE
        tokenizer_kwargs["cache_dir"] = _GGUF_CACHE_DIR

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_gguf_checkpoint_accepts_model_to_load()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs["cache_dir"] = _GGUF_CACHE_DIR

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                gguf_file=self.GGUF_FILE,
                cache_dir=_GGUF_CACHE_DIR,
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
            cache_dir=_GGUF_CACHE_DIR,
        )
        return self.config
