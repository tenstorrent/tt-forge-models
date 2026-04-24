# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Coder Next REAP GGUF model loader implementation for causal language modeling.
"""
import re

import numpy as np
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
    GGUFTensor,
    TensorProcessor,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)
from typing import Optional

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


def _patch_qwen3next_gguf_support():
    """Register qwen3next GGUF architecture support for transformers 5.x.

    The qwen3next architecture is a hybrid SSM-attention MoE model. Transformers 5.2.x
    has Qwen3NextForCausalLM but lacks GGUF loading support. This patch:
    - Registers qwen3next as a supported GGUF architecture
    - Maps GGUF config fields to Qwen3NextConfig fields
    - Adds a tensor processor to handle SSM-specific transformations
    - Patches load_gguf_checkpoint to set model_type and infer layer_types
    """
    if "qwen3next" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3next")

    qwen3_config = GGUF_TO_TRANSFORMERS_MAPPING["config"].get("qwen3", {})
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3next"] = dict(qwen3_config)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3next"].update(
        {
            "attention.key_length": "head_dim",
            "ssm.conv_kernel": "linear_conv_kernel_dim",
            "expert_count": "num_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_feed_forward_length": "moe_intermediate_size",
        }
    )

    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3next", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )
        # The GGUF file may report tokenizer class as 'qwen3_next' (with underscore)
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_next", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )

    class Qwen3NextTensorProcessor(TensorProcessor):
        def perform_fallback_tensor_mapping(
            self, gguf_to_hf_name_map, suffix, qual_name, hf_name
        ):
            # dt_bias is a bare Parameter (ends in .bias when split) but maps to
            # blk.N.ssm_dt (no suffix) in GGUF because gguf-py uses dt_proj not dt
            m = re.match(r"model\.layers\.(\d+)\.linear_attn\.dt_bias$", hf_name)
            if m:
                gguf_to_hf_name_map[f"blk.{m.group(1)}.ssm_dt"] = qual_name + hf_name

        def process(self, weights, name, **kwargs):
            if "ssm_conv1d.weight" in name:
                # conv1d.weight is (channels, 1, kernel_size) but GGUF stores (channels, kernel_size)
                weights = np.expand_dims(weights, axis=1)
            if "ffn_gate_inp_shexp" in name:
                # shared_expert_gate is (1, hidden_size) but quantized GGUF stores (hidden_size,)
                weights = np.expand_dims(weights, axis=0)
            return GGUFTensor(weights, name, {})

    _gguf_utils.TENSOR_PROCESSORS["qwen3next"] = Qwen3NextTensorProcessor


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    # gguf-py uses 'qwen3next' but transformers model_type is 'qwen3_next'
    effective_type = hf_model.config.model_type if model_type is None else model_type
    if effective_type == "qwen3_next":
        model_type = "qwen3next"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_qwen3next_gguf_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    config = result.get("config", {})
    if config.get("model_type") != "qwen3next":
        return result

    config["model_type"] = "qwen3_next"

    # Infer layer_types by reading which GGUF blocks have SSM tensors.
    # We must do this from the GGUF file because tensors may not be loaded yet
    # (return_tensors=False path used during config-only loading).
    if "layer_types" not in config:
        gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
        num_layers = config.get("num_hidden_layers", 48)
        layer_types = ["full_attention"] * num_layers
        if gguf_path is not None:
            try:
                from gguf import GGUFReader

                reader = GGUFReader(gguf_path)
                ssm_layers = set()
                for tensor in reader.tensors:
                    # SSM tensors like blk.N.ssm_in identify linear attention layers
                    if ".ssm_in" in tensor.name:
                        parts = tensor.name.split(".")
                        if len(parts) >= 2 and parts[1].isdigit():
                            ssm_layers.add(int(parts[1]))
                for i in range(num_layers):
                    if i in ssm_layers:
                        layer_types[i] = "linear_attention"
            except Exception:
                pass
        config["layer_types"] = layer_types

    if "shared_expert_intermediate_size" not in config:
        config["shared_expert_intermediate_size"] = config.get(
            "moe_intermediate_size", config.get("intermediate_size", 512)
        )

    return result


_patch_qwen3next_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Qwen3 Coder Next REAP GGUF model variants for causal language modeling."""

    QWEN_3_CODER_NEXT_REAP_GGUF = "REAP_GGUF"


class ModelLoader(ForgeModel):
    """Qwen3 Coder Next REAP GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_CODER_NEXT_REAP_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-Coder-Next-REAP-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_CODER_NEXT_REAP_GGUF

    GGUF_FILE = "Qwen3-Coder-Next-REAP.Q4_K_M.gguf"

    sample_text = "Write a Python function that checks if a number is prime."

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
            model="Qwen3 Coder Next REAP GGUF",
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
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
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
