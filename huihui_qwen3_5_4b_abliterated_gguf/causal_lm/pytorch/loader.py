# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen 3.5 4B Abliterated GGUF model loader implementation for causal language modeling.
"""
import contextlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.configuration_utils as _config_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer

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


def _register_qwen35_gguf_tables():
    """Register qwen35 GGUF architecture in transformers tables (idempotent)."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        GGUFTensor,
        TensorProcessor,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "ssm.conv_kernel": "linear_conv_kernel_dim",
        "ssm.state_size": None,
        "ssm.inner_size": None,
        "ssm.time_step_rank": None,
        "ssm.group_count": None,
        "full_attention_interval": "full_attention_interval",
        "vocab_size": "vocab_size",
    }

    class Qwen35TensorProcessor(TensorProcessor):
        def __init__(self, config=None):
            super().__init__(config=config)

        def process(self, weights, name, **kwargs):
            if "ssm_conv1d.weight" in name and weights.ndim == 2:
                weights = np.expand_dims(weights, axis=1)
            if "ssm_a" in name:
                weights = np.log(-weights)
            return GGUFTensor(weights, name, {})

    TENSOR_PROCESSORS["qwen35"] = Qwen35TensorProcessor

    GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUFQwen2Converter)
    GGUF_TO_FAST_CONVERTERS.setdefault("qwen3_5_text", GGUFQwen2Converter)


def _find_real_load_gguf_checkpoint():
    """Walk the patcher chain to find the real transformers load_gguf_checkpoint.

    Multiple loaders (bartowski, daniloreddy, mradermacher, etc.) install
    _patched_load_gguf_checkpoint that remaps qwen35->qwen3. We must bypass
    all of them and call the real function so our qwen35->qwen3_5_text remap
    takes effect.
    """
    seen = set()
    queue = [_gguf_utils.load_gguf_checkpoint]
    while queue:
        fn = queue.pop(0)
        fid = id(fn)
        if fid in seen:
            continue
        seen.add(fid)
        # The real function lives in modeling_gguf_pytorch_utils module
        if hasattr(fn, "__globals__") and fn.__globals__ is vars(_gguf_utils):
            return fn
        # Walk __globals__ for saved originals
        if hasattr(fn, "__globals__"):
            for name in ("_orig_load_gguf_checkpoint", "orig_load", "_real_load"):
                nxt = fn.__globals__.get(name)
                if callable(nxt) and id(nxt) not in seen:
                    queue.append(nxt)
        # Walk closure cells
        if hasattr(fn, "__closure__") and fn.__closure__:
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val) and id(val) not in seen:
                        queue.append(val)
                except ValueError:
                    pass
    # Fallback: return whatever is currently installed
    return _gguf_utils.load_gguf_checkpoint


def _build_qwen35_patcher(real_fn):
    """Return a load_gguf_checkpoint wrapper that remaps qwen35->qwen3_5_text."""

    def _patcher(*args, **kwargs):
        result = real_fn(*args, **kwargs)
        cfg = result.get("config", {})
        if cfg.get("model_type") == "qwen35":
            cfg["model_type"] = "qwen3_5_text"
            num_layers = cfg.get("num_hidden_layers", 32)
            interval = cfg.pop("full_attention_interval", 4)
            layer_types = []
            for i in range(num_layers):
                if (i + 1) % interval == 0:
                    layer_types.append("full_attention")
                else:
                    layer_types.append("linear_attention")
            cfg["layer_types"] = layer_types
        return result

    return _patcher


@contextlib.contextmanager
def _qwen35_load_ctx():
    """Temporarily install corrected load_gguf_checkpoint and get_gguf_hf_weights_map.

    Loaders imported before this one (bartowski, daniloreddy) and after
    (mradermacher) all remap qwen35->qwen3 which selects Qwen3ForCausalLM
    and causes weight mismatches. We bypass them via BFS to find the real
    transformers load_gguf_checkpoint, then wrap get_gguf_hf_weights_map
    to remap qwen3_5_text->qwen35 for the GGUF tensor-name lookup.
    """
    real_fn = _find_real_load_gguf_checkpoint()
    load_patcher = _build_qwen35_patcher(real_fn)

    # get_gguf_hf_weights_map: wrap the current chain (including onion008's patcher)
    # to remap qwen3_5_text -> qwen35 before calling the real function.
    _current_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _get_map_patcher(hf_model, processor, model_type=None, num_layers=None, **kwargs):
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type in ("qwen3_5_text", "qwen3_5"):
            model_type = "qwen35"
        return _current_get_map(hf_model, processor, model_type, num_layers, **kwargs)

    old_gguf = _gguf_utils.load_gguf_checkpoint
    old_cfg = _config_utils.load_gguf_checkpoint
    old_tok = _auto_tokenizer.load_gguf_checkpoint if hasattr(_auto_tokenizer, "load_gguf_checkpoint") else None
    old_get_map = _gguf_utils.get_gguf_hf_weights_map

    _gguf_utils.load_gguf_checkpoint = load_patcher
    _config_utils.load_gguf_checkpoint = load_patcher
    if old_tok is not None:
        _auto_tokenizer.load_gguf_checkpoint = load_patcher
    _gguf_utils.get_gguf_hf_weights_map = _get_map_patcher
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = old_gguf
        _config_utils.load_gguf_checkpoint = old_cfg
        if old_tok is not None:
            _auto_tokenizer.load_gguf_checkpoint = old_tok
        _gguf_utils.get_gguf_hf_weights_map = old_get_map


# Register GGUF tables at import time
_register_qwen35_gguf_tables()


class ModelVariant(StrEnum):
    """Available Huihui Qwen 3.5 4B Abliterated GGUF model variants for causal language modeling."""

    HUIHUI_QWEN3_5_4B_ABLITERATED_GGUF = "4B_Abliterated_GGUF"


class ModelLoader(ForgeModel):
    """Huihui Qwen 3.5 4B Abliterated GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_5_4B_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3.5-4B-abliterated-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_5_4B_ABLITERATED_GGUF

    GGUF_FILE = "Huihui-Qwen3.5-4B-abliterated.i1-Q4_K_M.gguf"

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
            model="Huihui Qwen 3.5 4B Abliterated GGUF",
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
        # use_cache=False avoids Qwen3_5DynamicCache in model output
        model_kwargs.setdefault("use_cache", False)

        if self.num_layers is not None:
            with _qwen35_load_ctx():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _qwen35_load_ctx():
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "up_proj"):
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
                attn = layer.self_attn
                shard_specs[attn.q_proj.weight] = ("model", "batch")
                if hasattr(attn.q_proj, "bias") and attn.q_proj.bias is not None:
                    shard_specs[attn.q_proj.bias] = ("model",)
                shard_specs[attn.k_proj.weight] = ("model", "batch")
                if hasattr(attn.k_proj, "bias") and attn.k_proj.bias is not None:
                    shard_specs[attn.k_proj.bias] = ("model",)
                shard_specs[attn.v_proj.weight] = ("model", "batch")
                if hasattr(attn.v_proj, "bias") and attn.v_proj.bias is not None:
                    shard_specs[attn.v_proj.bias] = ("model",)
                shard_specs[attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        with _qwen35_load_ctx():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
