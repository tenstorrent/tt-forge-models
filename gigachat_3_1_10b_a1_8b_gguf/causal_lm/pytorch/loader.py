# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GigaChat 3.1 10B A1.8B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel


def _patch_transformers_deepseek_v2_gguf():
    """Monkey-patch transformers to add deepseek_v2 GGUF architecture support.

    The GigaChat 3.1 GGUF file uses 'deepseek_v2' as the architecture name,
    which transformers does not yet register in its GGUF loading infrastructure.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "deepseek_v2" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    GGUF_SUPPORTED_ARCHITECTURES.append("deepseek_v2")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["deepseek_v2"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "rope.dimension_count": "qk_rope_head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": None,
        "attention.value_length": None,
        "attention.key_length_mla": "qk_nope_head_dim",
        "attention.value_length_mla": "v_head_dim",
        "attention.q_lora_rank": "q_lora_rank",
        "attention.kv_lora_rank": "kv_lora_rank",
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_count": "n_shared_experts",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_weights_scale": "routed_scaling_factor",
        "expert_weights_norm": "norm_topk_prob",
        "leading_dense_block_count": "first_k_dense_replace",
        "expert_feed_forward_length": "moe_intermediate_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "deepseek_v2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter

    # deepseek_v2 model_type is not in gguf-py's MODEL_ARCH_NAMES; map it to deepseek2
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    _orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if (
            model_type is None
            and getattr(getattr(hf_model, "config", None), "model_type", None)
            == "deepseek_v2"
        ):
            model_type = "deepseek2"
        elif model_type == "deepseek_v2":
            model_type = "deepseek2"
        return _orig_get_gguf_hf_weights_map(
            hf_model, processor, model_type, num_layers, qual_name
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
    import transformers.modeling_utils as _modeling_utils

    if hasattr(_modeling_utils, "get_gguf_hf_weights_map"):
        _modeling_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_transformers_deepseek_v2_gguf()
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
    """Available GigaChat 3.1 10B A1.8B GGUF model variants for causal language modeling."""

    GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF = "GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """GigaChat 3.1 10B A1.8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="ai-sage/GigaChat3.1-10B-A1.8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF

    GGUF_FILE = "GigaChat3.1-10B-A1.8B-q4_K_M.gguf"

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
            model="GigaChat 3.1 10B A1.8B GGUF",
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
