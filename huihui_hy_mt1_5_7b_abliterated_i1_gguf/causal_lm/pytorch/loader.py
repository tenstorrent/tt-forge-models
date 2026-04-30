# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui HY-MT1.5 7B abliterated i1 GGUF model loader implementation for causal language modeling.
"""
import threading
from typing import Optional

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    LlamaTensorProcessor,
    TENSOR_PROCESSORS,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.hunyuan_v1_dense import HunYuanDenseV1Config

# Thread-local used to pass model_to_load through broken wrapper chains.
# Broken wrappers drop model_to_load; we store it here in _fixed_load so
# _patched_get_gguf_hf_weights_map can recover it after the chain drops it.
_tls = threading.local()

_HUNYUAN_DENSE_GGUF_ARCH = "hunyuan-dense"
_HUNYUAN_DENSE_MODEL_TYPE = "hunyuan_v1_dense"

_HUNYUAN_DENSE_CONFIG_MAP = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "attention.key_length": "head_dim",
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
}


def _patch_hunyuan_dense_gguf():
    """Register hunyuan-dense GGUF architecture for HunYuanDenseV1ForCausalLM.

    Avoids wrapping load_gguf_checkpoint entirely (which causes TypeError chain
    issues in pytest sessions with multiple loaders patching it). Instead, register
    all required tables and add an AutoConfig alias for the hyphenated arch name.
    """
    if _HUNYUAN_DENSE_GGUF_ARCH in GGUF_SUPPORTED_ARCHITECTURES:
        return  # idempotent

    GGUF_SUPPORTED_ARCHITECTURES.append(_HUNYUAN_DENSE_GGUF_ARCH)

    cfg_mapping = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]
    cfg_mapping[_HUNYUAN_DENSE_GGUF_ARCH] = _HUNYUAN_DENSE_CONFIG_MAP

    TENSOR_PROCESSORS[_HUNYUAN_DENSE_GGUF_ARCH] = LlamaTensorProcessor

    # Tokenizer: keyed by GGUF arch name, not model_type
    if _HUNYUAN_DENSE_GGUF_ARCH not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS[_HUNYUAN_DENSE_GGUF_ARCH] = GGUFGPTConverter

    # Register "hunyuan-dense" as a CONFIG_MAPPING alias so AutoConfig.from_pretrained
    # resolves HunYuanDenseV1Config when model_type=="hunyuan-dense" in the GGUF config
    # dict. Use CONFIG_MAPPING.register (not AutoConfig.register) because the latter
    # rejects a config whose model_type attribute doesn't match the registered key.
    if _HUNYUAN_DENSE_GGUF_ARCH not in CONFIG_MAPPING:
        CONFIG_MAPPING.register(_HUNYUAN_DENSE_GGUF_ARCH, HunYuanDenseV1Config)


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Map hunyuan_v1_dense (HF model_type) back to hunyuan-dense (GGUF arch name).

    Also recovers hf_model from the thread-local set by _fixed_load when broken
    wrappers in the chain have dropped the model_to_load kwarg.
    """
    effective_model = hf_model
    if effective_model is None:
        effective_model = getattr(_tls, "model_to_load", None)
    resolved_type = (
        model_type
        if model_type is not None
        else getattr(getattr(effective_model, "config", None), "model_type", None)
    )
    if resolved_type == _HUNYUAN_DENSE_MODEL_TYPE:
        model_type = _HUNYUAN_DENSE_GGUF_ARCH
    return _orig_get_gguf_hf_weights_map(
        effective_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_patch_hunyuan_dense_gguf()
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


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


class ModelVariant(StrEnum):
    """Available Huihui HY-MT1.5 7B abliterated i1 GGUF model variants for causal language modeling."""

    HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF = "HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF"


class ModelLoader(ForgeModel):
    """Huihui HY-MT1.5 7B abliterated i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-HY-MT1.5-7B-abliterated-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF

    GGUF_FILE = "Huihui-HY-MT1.5-7B-abliterated.i1-Q4_K_M.gguf"

    sample_text = (
        "Translate the following segment into Chinese: The weather is nice today."
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
            model="Huihui HY-MT1.5 7B abliterated i1 GGUF",
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

        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        if getattr(config, "head_dim", None) is None:
            config.head_dim = config.hidden_size // config.num_attention_heads
        model_kwargs["config"] = config

        # Install an outermost wrapper that accepts model_to_load (transformers
        # 5.x kwarg) and stores it in thread-local storage.  Broken wrappers in
        # the chain drop model_to_load; _patched_get_gguf_hf_weights_map
        # recovers it from _tls so the real function can still build the weight
        # key mapping with the actual dummy_model.
        _saved_fn = _gguf_utils.load_gguf_checkpoint

        def _fixed_load(path, return_tensors=False, model_to_load=None, **kw):
            _tls.model_to_load = model_to_load
            try:
                return _saved_fn(path, return_tensors=return_tensors)
            finally:
                _tls.model_to_load = None

        _gguf_utils.load_gguf_checkpoint = _fixed_load
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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
