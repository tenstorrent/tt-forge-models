# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UnifiedReward-Flex-qwen35-27b i1 GGUF model loader for causal language modeling.
"""

from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


def _patch_transformers_qwen35_gguf():
    """Monkey-patch transformers to add qwen35 GGUF architecture support.

    Transformers 5.x has Qwen3_5ForCausalLM but lacks GGUF loading support
    for the qwen35 architecture. The gguf library (>=0.18) already knows about
    qwen35 tensor names, so we only need to bridge transformers' config/tensor
    processing layer.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        GGUFTensor,
        TensorProcessor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    if "qwen35" not in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
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

    if "qwen35" not in TENSOR_PROCESSORS:

        class Qwen35TensorProcessor(TensorProcessor):
            def __init__(self, config=None):
                super().__init__(config=config)

            def preprocess_name(self, hf_name: str) -> str:
                return hf_name.replace(".dt_bias", ".dt_proj")

            def process(self, weights, name, **kwargs):
                if "ssm_conv1d.weight" in name:
                    if weights.ndim == 2:
                        weights = np.expand_dims(weights, axis=1)
                if "ssm_a" in name:
                    weights = np.log(-weights)
                return GGUFTensor(weights, name, {})

        TENSOR_PROCESSORS["qwen35"] = Qwen35TensorProcessor

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    for key in ("qwen35", "qwen3_5_text"):
        if key not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS[key] = GGUFQwen2Converter

    # Patch load_gguf_checkpoint to handle qwen35 -> qwen3_5_text
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen35":
            result["config"]["model_type"] = "qwen3_5_text"
            config = result["config"]
            num_layers = config.get("num_hidden_layers", 32)
            interval = config.pop("full_attention_interval", 4)
            layer_types = []
            for i in range(num_layers):
                if (i + 1) % interval == 0:
                    layer_types.append("full_attention")
                else:
                    layer_types.append("linear_attention")
            config["layer_types"] = layer_types
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Patch get_gguf_hf_weights_map to handle qwen3_5_text -> qwen35
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_text", "qwen3_5"):
            model_type = "qwen35"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_qwen35_gguf()


class ModelVariant(StrEnum):
    """Available UnifiedReward-Flex-qwen35-27b i1 GGUF model variants."""

    UNIFIED_REWARD_FLEX_QWEN35_27B_I1_Q4_K_M_GGUF = (
        "UnifiedReward_Flex_qwen35_27b_i1_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """UnifiedReward-Flex-qwen35-27b i1 GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.UNIFIED_REWARD_FLEX_QWEN35_27B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/UnifiedReward-Flex-qwen35-27b-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNIFIED_REWARD_FLEX_QWEN35_27B_I1_Q4_K_M_GGUF

    GGUF_FILE = "UnifiedReward-Flex-qwen35-27b.i1-Q4_K_M.gguf"

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
            model="UnifiedReward-Flex-qwen35-27b i1 GGUF",
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
