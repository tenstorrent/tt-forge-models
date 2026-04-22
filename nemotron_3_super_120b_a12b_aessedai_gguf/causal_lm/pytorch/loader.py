# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_nemotron_h_moe_gguf():
    """Monkey-patch transformers to add nemotron_h_moe GGUF architecture support.

    Transformers 5.x has NemotronHForCausalLM but lacks GGUF loading support
    for the nemotron_h_moe architecture. The nemotron_h_moe GGUF format stores
    per-layer config (kv_heads, ffn sizes) as lists from which we infer
    layers_block_type (mamba / attention / moe).
    """
    import numpy as np
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TensorProcessor,
        GGUFTensor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["nemotron_h_moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": None,
        "embedding_length": "hidden_size",
        "vocab_size": "vocab_size",
        "rope.dimension_count": None,
        "rope.freq_base": None,
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": None,
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        "feed_forward_length": None,
        "expert_count": "n_routed_experts",
        "expert_shared_count": "n_shared_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_weights_norm": "norm_topk_prob",
        "expert_weights_scale": "routed_scaling_factor",
        "moe_latent_size": "moe_latent_size",
        "ssm.conv_kernel": "conv_kernel",
        "ssm.state_size": "ssm_state_size",
        "ssm.group_count": "n_groups",
        "ssm.time_step_rank": "mamba_num_heads",
        "ssm.inner_size": None,
    }

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

    if "nemotron_h_moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUFGPTConverter

    class NemotronHMoeTensorProcessor(TensorProcessor):
        def process(self, weights, name, **kwargs):
            if "ssm_conv1d.weight" in name:
                weights = np.expand_dims(weights, axis=1)
            if "ssm_a" in name:
                weights = np.log(-weights)
            return GGUFTensor(weights, name, {})

    gguf_utils.TENSOR_PROCESSORS["nemotron_h_moe"] = NemotronHMoeTensorProcessor

    _orig_load_gguf_checkpoint = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load_gguf_checkpoint(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") != "nemotron_h_moe":
            return result

        config["model_type"] = "nemotron_h"

        gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
        if gguf_path:
            try:
                from gguf import GGUFReader
                from transformers.modeling_gguf_pytorch_utils import read_field

                reader = GGUFReader(gguf_path)
                kv_heads = read_field(reader, "nemotron_h_moe.attention.head_count_kv")
                ffn_sizes = read_field(reader, "nemotron_h_moe.feed_forward_length")
                inner_size = read_field(reader, "nemotron_h_moe.ssm.inner_size")

                if kv_heads and ffn_sizes:
                    layers_block_type = []
                    for kv, ffn in zip(kv_heads, ffn_sizes):
                        if kv > 0:
                            layers_block_type.append("attention")
                        elif ffn > 0:
                            layers_block_type.append("moe")
                        else:
                            layers_block_type.append("mamba")
                    config["layers_block_type"] = layers_block_type

                num_heads = config.get("mamba_num_heads")
                if num_heads and inner_size:
                    config["mamba_head_dim"] = inner_size[0] // num_heads
            except Exception:
                pass

        kv = config.get("num_key_value_heads")
        if isinstance(kv, list):
            non_zero = [h for h in kv if h > 0]
            config["num_key_value_heads"] = max(non_zero) if non_zero else 8

        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as _tok_auto
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_utils as _modeling_utils

    for _mod in (_tok_auto, _config_utils, _modeling_utils):
        if hasattr(_mod, "load_gguf_checkpoint"):
            _mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    _orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "nemotron_h":
            model_type = "nemotron_h_moe"
        return _orig_get_weights_map(
            hf_model, processor, model_type, num_layers, qual_name
        )

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_transformers_nemotron_h_moe_gguf()


class ModelVariant(StrEnum):
    """Available AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF = "3_Super_120B_A12B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF

    GGUF_FILE = (
        "Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M-00001-of-00003.gguf"
    )

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
            model="Nemotron 3 Super 120B A12B AesSedai GGUF",
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
