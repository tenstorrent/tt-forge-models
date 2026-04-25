# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling.
"""
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
    GGUFTensor,
    TensorProcessor,
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


class _NemotronHMoeTensorProcessor(TensorProcessor):
    """Tensor processor for NemotronH MoE GGUF checkpoints."""

    def process(self, weights, name, **kwargs):
        # RMS norm weights use offset convention (store weight - 1, HF expects raw weight)
        if "norm.weight" in name:
            weights = weights - 1
        # SSM A parameter: GGUF stores negative values; HF expects log(-A) = A_log
        if "ssm_a" in name and "norm" not in name:
            weights = np.log(-weights)
        # SSM conv1d: GGUF [kernel_size, channels] → HF [channels, 1, kernel_size]
        if "ssm_conv1d.weight" in name:
            weights = np.expand_dims(weights, axis=1)
        # Expert weights: GGUF (dim_a, dim_b, num_experts) → HF (num_experts, dim_a, dim_b)
        if (
            "ffn_down_exps.weight" in name or "ffn_up_exps.weight" in name
        ) and weights.ndim == 3:
            weights = np.transpose(weights, (2, 1, 0))
        return GGUFTensor(weights, name, {})


def _patch_nemotron_h_moe_support():
    """Register nemotron_h_moe GGUF architecture, mapping it to the nemotron_h model type.

    The GGUF file uses architecture 'nemotron_h_moe' but transformers implements
    this as the 'nemotron_h' model type with layers_block_type derived from per-layer
    metadata stored in the GGUF fields.
    """
    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["nemotron_h_moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": None,
        "attention.head_count": "num_attention_heads",
        # Per-layer list; post-processed to scalar + used for layers_block_type derivation
        "attention.head_count_kv": "_kv_heads_per_layer",
        "attention.key_length": "head_dim",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        "attention.layer_norm_rms_epsilon": None,
        "attention.value_length": None,
        # Per-layer list; post-processed to derive layers_block_type
        "feed_forward_length": "_ffn_per_layer",
        "expert_count": "n_routed_experts",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_shared_count": "n_shared_experts",
        "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
        "expert_used_count": "num_experts_per_tok",
        "expert_weights_norm": "norm_topk_prob",
        "moe_latent_size": "moe_latent_size",
        "ssm.conv_kernel": "conv_kernel",
        "ssm.group_count": "n_groups",
        "ssm.state_size": "ssm_state_size",
        "vocab_size": "vocab_size",
    }

    _gguf_utils.TENSOR_PROCESSORS["nemotron_h_moe"] = _NemotronHMoeTensorProcessor

    if "nemotron" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUF_TO_FAST_CONVERTERS["nemotron"]


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add nemotron_h_moe GGUF architecture support."""
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )

    config = result.get("config", {})
    if config.get("model_type") == "nemotron_h_moe":
        config["model_type"] = "nemotron_h"
        # Disable Mamba CUDA kernels; use pure-PyTorch fallback
        config["use_mamba_kernels"] = False

        # Retrieve and remove temporary per-layer fields
        kv_heads = config.pop("_kv_heads_per_layer", 8)
        ffn_lengths = config.pop("_ffn_per_layer", [])

        if isinstance(kv_heads, list):
            non_zero_kv = [h for h in kv_heads if h > 0]
            config["num_key_value_heads"] = max(non_zero_kv) if non_zero_kv else 8

            # Derive layers_block_type: attention where kv_heads>0, moe where ffn>0, else mamba
            if isinstance(ffn_lengths, list) and len(ffn_lengths) == len(kv_heads):
                ffn_iter = ffn_lengths
            else:
                ffn_iter = [0] * len(kv_heads)

            layers_block_type = []
            for kv, ff in zip(kv_heads, ffn_iter):
                if kv > 0:
                    layers_block_type.append("attention")
                elif ff > 0:
                    layers_block_type.append("moe")
                else:
                    layers_block_type.append("mamba")
            config["layers_block_type"] = layers_block_type
        else:
            config["num_key_value_heads"] = kv_heads

        # Remove num_hidden_layers: NemotronHConfig derives it from layers_block_type
        config.pop("num_hidden_layers", None)

    return result


_patch_nemotron_h_moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Nemotron 3 Super 120B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF = (
        "3_Super_120B_A12B_BF16_heretic_i1_GGUF"
    )
    GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF = "ggml_org_3_Super_120B_GGUF"


class ModelLoader(ForgeModel):
    """Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: LLMModelConfig(
            pretrained_model_name="ggml-org/Nemotron-3-Super-120B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: "NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic.i1-Q4_K_M.gguf",
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: "Nemotron-3-Super-120B-Q4_K.gguf",
    }

    sample_text = "Give me a short introduction to large language model."

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

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
            model="Nemotron 3 Super 120B GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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

        messages = [{"role": "user", "content": self.sample_text}]
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
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
