# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Daniloreddy Qwen 3.5 2B GGUF model loader implementation for causal language modeling.
"""
import contextlib
import numpy as np
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


def _register_qwen35_gguf_support():
    """Register qwen35 in GGUF architecture/config/tensor mappings at import time."""
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        GGUFTensor,
        TensorProcessor,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter

    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    GGUF_TO_TRANSFORMERS_MAPPING.setdefault("config", {})["qwen35"] = {
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

    if "qwen35" not in TENSOR_PROCESSORS:
        class _Qwen35TensorProcessor(TensorProcessor):
            def process(self, weights, name, **kwargs):
                if "ssm_conv1d.weight" in name and weights.ndim == 2:
                    weights = np.expand_dims(weights, axis=1)
                if "ssm_a" in name:
                    weights = np.log(-weights)
                return GGUFTensor(weights, name, {})
        TENSOR_PROCESSORS["qwen35"] = _Qwen35TensorProcessor

    GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUFQwen2Converter)


def _make_qwen35_load_fn(orig_fn):
    """Return a load_gguf_checkpoint wrapper that remaps qwen35 → qwen3_5_text."""
    def _patched(*args, **kwargs):
        result = orig_fn(*args, **kwargs)
        cfg = result.get("config", {})
        if cfg.get("model_type") == "qwen35":
            cfg["model_type"] = "qwen3_5_text"
            num_layers = cfg.get("num_hidden_layers", 24)
            interval = cfg.pop("full_attention_interval", 4)
            cfg["layer_types"] = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(num_layers)
            ]
        return result
    return _patched


@contextlib.contextmanager
def _qwen35_load_context():
    """Temporarily install the correct load_gguf_checkpoint for qwen35."""
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    orig = gguf_utils.load_gguf_checkpoint
    gguf_utils.load_gguf_checkpoint = _make_qwen35_load_fn(orig)
    try:
        yield
    finally:
        gguf_utils.load_gguf_checkpoint = orig


_register_qwen35_gguf_support()


class ModelVariant(StrEnum):
    """Available Daniloreddy Qwen 3.5 2B GGUF model variants for causal language modeling."""

    DANILOREDDY_QWEN_3_5_2B_GGUF = "2B_GGUF"


class ModelLoader(ForgeModel):
    """Daniloreddy Qwen 3.5 2B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DANILOREDDY_QWEN_3_5_2B_GGUF: LLMModelConfig(
            pretrained_model_name="daniloreddy/Qwen3.5-2B_GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DANILOREDDY_QWEN_3_5_2B_GGUF

    GGUF_FILE = "Qwen3.5-2B_Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="Daniloreddy Qwen 3.5 2B GGUF",
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

        with _qwen35_load_context():
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
