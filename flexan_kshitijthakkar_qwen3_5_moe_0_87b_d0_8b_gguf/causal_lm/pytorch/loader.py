# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flexan Kshitijthakkar Qwen3.5 MoE 0.87B d0.8B GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_qwen35moe_gguf():
    """Monkey-patch transformers to add qwen35moe GGUF architecture support.

    Transformers 5.x has Qwen3_5MoeForCausalLM but lacks GGUF loading support
    for the qwen35moe architecture. We bridge the gap by registering qwen35moe
    config/tensor mappings and converting the model_type to qwen3_5_moe_text.

    Also adds fallback mappings for GGUF files that store expert weights as
    separate ffn_gate_exps/ffn_up_exps tensors (rather than merged
    ffn_gate_up_exps), ensuring they map correctly to the HF model's merged
    gate_up_proj parameter.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen35moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen35moe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

    # 2. Add config mapping for qwen35moe (based on qwen3_moe + Qwen3.5 fields)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "full_attention_interval": "full_attention_interval",
        # MoE FFN sizes (distinct from shared expert)
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
        # GatedDeltaNet (linear attention) SSM parameters
        "ssm.group_count": "linear_num_value_heads",
        "ssm.state_size": "linear_value_head_dim",
        "ssm.conv_kernel": "linear_conv_kernel_dim",
    }

    # 3. Register a custom tensor processor for qwen35moe that handles SSM conv1d reshape
    import numpy as np
    from transformers.modeling_gguf_pytorch_utils import (
        GGUFTensor,
        Qwen2MoeTensorProcessor,
    )

    class Qwen35MoeTensorProcessor(Qwen2MoeTensorProcessor):
        def process(self, weights, name, **kwargs):
            if "ssm_conv1d.weight" in name:
                # GGUF stores conv1d as [conv_dim, kernel_size]; HF Conv1d expects [conv_dim, 1, kernel_size]
                weights = np.expand_dims(weights, axis=1)
                return GGUFTensor(weights, name, {})
            return super().process(weights, name, **kwargs)

    TENSOR_PROCESSORS["qwen35moe"] = Qwen35MoeTensorProcessor

    # 4. Register tokenizer converter
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen35moe"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        GGUF_TO_FAST_CONVERTERS["qwen3_5_moe_text"] = GGUF_TO_FAST_CONVERTERS[
            "qwen3_moe"
        ]

    # 5. Patch load_gguf_checkpoint to handle qwen35moe -> qwen3_5_moe_text
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen35moe":
            result["config"]["model_type"] = "qwen3_5_moe_text"
            # Generate layer_types from full_attention_interval
            config = result["config"]
            num_layers = config.get("num_hidden_layers", 40)
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

    # Also patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map to handle qwen3_5_moe_text -> qwen35moe
    # and add fallback mappings for separate gate/up expert tensors.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
            model_type = "qwen35moe"
        result = orig_get_map(hf_model, processor, model_type, num_layers, qual_name)
        # Some qwen35moe GGUF files store separate ffn_gate_exps and ffn_up_exps
        # tensors instead of the merged ffn_gate_up_exps. Add fallback aliases so
        # Qwen2MoeTensorProcessor can map them to the HF model's gate_up_proj.
        if model_type == "qwen35moe" and qual_name == "":
            num_layers_val = (
                num_layers
                if num_layers is not None
                else hf_model.config.num_hidden_layers
            )
            for bid in range(num_layers_val):
                merged_key = f"blk.{bid}.ffn_gate_up_exps"
                if merged_key in result:
                    result.setdefault(f"blk.{bid}.ffn_gate_exps", result[merged_key])
                    result.setdefault(f"blk.{bid}.ffn_up_exps", result[merged_key])
        return result

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available Flexan Kshitijthakkar Qwen3.5 MoE 0.87B d0.8B GGUF model variants for causal language modeling."""

    FLEXAN_KSHITIJTHAKKAR_QWEN3_5_MOE_0_87B_D0_8B_GGUF = "MoE_0.87B_d0.8B_GGUF"


class ModelLoader(ForgeModel):
    """Flexan Kshitijthakkar Qwen3.5 MoE 0.87B d0.8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FLEXAN_KSHITIJTHAKKAR_QWEN3_5_MOE_0_87B_D0_8B_GGUF: LLMModelConfig(
            pretrained_model_name="Flexan/kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLEXAN_KSHITIJTHAKKAR_QWEN3_5_MOE_0_87B_D0_8B_GGUF

    GGUF_FILE = "qwen3.5-moe-0.87B-d0.8B.Q4_K_M.gguf"

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
            model="Flexan Kshitijthakkar Qwen3.5 MoE 0.87B d0.8B GGUF",
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
