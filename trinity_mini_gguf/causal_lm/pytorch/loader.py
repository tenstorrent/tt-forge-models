# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Trinity Mini GGUF model loader implementation for causal language modeling.
"""
import re

import numpy as np
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
    TensorProcessor,
    GGUFTensor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
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

_AFMOE_GGUF_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.key_length": "head_dim",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
    "expert_count": "num_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_shared_count": "num_shared_experts",
    "expert_feed_forward_length": "moe_intermediate_size",
    "leading_dense_block_count": "num_dense_layers",
    "attention.sliding_window": "sliding_window",
    "expert_group_count": "n_group",
    "expert_group_used_count": "topk_group",
    "expert_weights_scale": "route_scale",
    "expert_weights_norm": "route_norm",
}

_EXPERT_BATCH_PATTERN = re.compile(
    r"blk\.(?P<bid>\d+)\.ffn_(?P<w>gate|up|down)_exps(?P<suffix>\.weight)?$"
)


class _AfmoeTensorProcessor(TensorProcessor):
    """Splits batched GGUF expert tensors into individual HF expert parameters.

    The afmoe GGUF format stores all expert weights as a single 3-D tensor
    (num_experts, out_features, in_features) while the HF AfmoeModel keeps
    each expert as a separate nn.Linear in a ModuleList.
    """

    _EXPERT_IDX_PATTERN = re.compile(r"(mlp\.experts)\.\d+\.")

    def __init__(self, config=None):
        super().__init__(config=config)
        self.num_experts = (config or {}).get("num_experts", 128)

    def preprocess_name(self, hf_name: str) -> str:
        # Strip expert index so get_tensor_name_map can resolve the batched GGUF name.
        # e.g. model.layers.2.mlp.experts.0.gate_proj -> model.layers.2.mlp.experts.gate_proj
        return self._EXPERT_IDX_PATTERN.sub(r"\1.", hf_name)

    def process(self, weights, name, **kwargs):
        if m := _EXPERT_BATCH_PATTERN.match(name):
            parsed_parameters = kwargs.get("parsed_parameters", {})
            bid = m["bid"]
            w = m["w"]  # gate | up | down
            # dequantized shape: (num_experts, out_features, in_features)
            for i in range(self.num_experts):
                hf_name = f"model.layers.{bid}.mlp.experts.{i}.{w}_proj.weight"
                parsed_parameters["tensors"][hf_name] = torch.from_numpy(
                    np.copy(weights[i])
                )
            # Signal to the outer loop to skip normal tensor insertion.
            return GGUFTensor(weights, None, {})
        return GGUFTensor(weights, name, {})


def _patch_afmoe_support():
    """Register afmoe as a supported GGUF architecture.

    arcee-ai/Trinity-Mini uses a custom afmoe architecture not yet
    recognised by transformers.  We register it here so the GGUF loading
    machinery can parse the config and tokenizer, and supply a custom
    TensorProcessor that splits the batched expert weights.
    """
    if "afmoe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("afmoe")

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "afmoe", _AFMOE_GGUF_CONFIG_MAPPING
    )

    # afmoe uses a GPT-2-style (BPE) tokenizer in the GGUF file.
    if "gpt2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("afmoe", GGUF_TO_FAST_CONVERTERS["gpt2"])

    # Register the custom TensorProcessor so batched expert weights are split.
    _gguf_utils.TENSOR_PROCESSORS.setdefault("afmoe", _AfmoeTensorProcessor)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    _patch_afmoe_support()
    return _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )


_patch_afmoe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Trinity Mini GGUF model variants for causal language modeling."""

    TRINITY_MINI_GGUF = "GGUF"
    TRINITY_MINI_I1_GGUF = "i1_GGUF"


class ModelLoader(ForgeModel):
    """Trinity Mini GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TRINITY_MINI_GGUF: LLMModelConfig(
            pretrained_model_name="MaziyarPanahi/Trinity-Mini-GGUF",
            max_length=128,
        ),
        ModelVariant.TRINITY_MINI_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Trinity-Mini-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRINITY_MINI_GGUF

    _GGUF_FILES = {
        ModelVariant.TRINITY_MINI_GGUF: "Trinity-Mini.Q4_K_M.gguf",
        ModelVariant.TRINITY_MINI_I1_GGUF: "Trinity-Mini.i1-Q4_K_M.gguf",
    }

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
            model="Trinity Mini GGUF",
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
        tokenizer_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]
        tokenizer_kwargs["trust_remote_code"] = True

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
        model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]
        model_kwargs["trust_remote_code"] = True

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                gguf_file=self._GGUF_FILES[self._variant],
                trust_remote_code=True,
            )
            config.num_hidden_layers = self.num_layers
            if hasattr(config, "layer_types"):
                config.layer_types = config.layer_types[: self.num_layers]
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
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                for expert in mlp.experts:
                    shard_specs[expert.gate_proj.weight] = ("model", "batch")
                    shard_specs[expert.up_proj.weight] = ("model", "batch")
                    shard_specs[expert.down_proj.weight] = ("batch", "model")
            if hasattr(mlp, "shared_experts") and mlp.shared_experts is not None:
                shard_specs[mlp.shared_experts.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_experts.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_experts.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
            trust_remote_code=True,
        )
        return self.config
