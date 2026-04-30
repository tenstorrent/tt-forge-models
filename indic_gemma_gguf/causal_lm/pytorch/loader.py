# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Indic Gemma GGUF model loader implementation for causal language modeling.
"""
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


def _patch_gemma_v1_support():
    # transformers 5.x dropped gemma v1 from GGUF_CONFIG_MAPPING; re-register it
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        Gemma2TensorProcessor,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGemmaConverter

    _gemma_config = {
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
        "vocab_size": "vocab_size",
    }
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "gemma2" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "gemma", _gemma_config
            )
    if "gemma" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("gemma")
    _gguf_utils.TENSOR_PROCESSORS.setdefault("gemma", Gemma2TensorProcessor)
    GGUF_TO_FAST_CONVERTERS.setdefault("gemma", GGUFGemmaConverter)


_patch_gemma_v1_support()


def _get_real_load_gguf_checkpoint():
    """Walk the patch chain to find the original transformers load_gguf_checkpoint.

    Other loaders install narrow-sig wrappers (gguf_path, return_tensors=False)
    at import time, capturing the prior value in either module globals
    (_orig_load_gguf_checkpoint) or a closure cell (orig_load). Use a recursive
    DFS over both to reach the original transformers function.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    visited: set = set()

    def _is_real(f) -> bool:
        return (
            getattr(f, "__qualname__", "") == "load_gguf_checkpoint"
            and getattr(f, "__module__", "") == "transformers.modeling_gguf_pytorch_utils"
        )

    def _search(f):
        fid = id(f)
        if fid in visited:
            return None
        visited.add(fid)
        if _is_real(f):
            return f
        # Closures first: handles orig_load = ... / def patch(*args, **kwargs) pattern
        if getattr(f, "__closure__", None):
            for cell in f.__closure__:
                try:
                    v = cell.cell_contents
                    if callable(v):
                        r = _search(v)
                        if r is not None:
                            return r
                except Exception:
                    pass
        # Module globals: handles _orig_load_gguf_checkpoint = ... pattern
        for name, v in getattr(f, "__globals__", {}).items():
            if callable(v) and "orig" in name.lower() and id(v) not in visited:
                r = _search(v)
                if r is not None:
                    return r
        return None

    result = _search(_gguf_utils.load_gguf_checkpoint)
    return result if result is not None else _gguf_utils.load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Indic Gemma GGUF model variants for causal language modeling."""

    INDIC_GEMMA_7B_SFT_NAVARASA_2_0_GGUF = "7B_SFT_NAVARASA_2.0_GGUF"


class ModelLoader(ForgeModel):
    """Indic Gemma GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.INDIC_GEMMA_7B_SFT_NAVARASA_2_0_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Indic-gemma-7b-finetuned-sft-Navarasa-2.0-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INDIC_GEMMA_7B_SFT_NAVARASA_2_0_GGUF

    GGUF_FILE = "Indic-gemma-7b-finetuned-sft-Navarasa-2.0.i1-Q4_K_M.gguf"

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
            model="Indic Gemma GGUF",
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
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils

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

        # Other loaders install narrow-sig (gguf_path, return_tensors=False) wrappers
        # at import time. Restore the real function just before from_pretrained so
        # the model_to_load kwarg added in transformers 5.2 is not rejected.
        _real = _get_real_load_gguf_checkpoint()
        _prev = _gguf_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _real
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _prev

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": self.sample_text}]
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
