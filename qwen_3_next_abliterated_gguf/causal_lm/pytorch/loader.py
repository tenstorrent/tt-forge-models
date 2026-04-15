# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 Next Abliterated GGUF model loader implementation for causal language modeling.
"""
import re

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


def _patch_transformers_qwen3next_gguf():
    """Monkey-patch transformers to add qwen3next GGUF architecture support.

    Transformers 5.x has Qwen3NextForCausalLM but lacks GGUF loading support
    for the qwen3next architecture. The gguf library (>=0.18) already knows about
    qwen3next tensor names, so we only need to bridge transformers' config/tensor
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

    if "qwen3next" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3next as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3next")

    # 2. Add config mapping for qwen3next
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3next"] = {
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
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
        "ssm.conv_kernel": "linear_conv_kernel_dim",
        "ssm.state_size": None,
        "ssm.inner_size": None,
        "ssm.time_step_rank": None,
        "ssm.group_count": None,
    }

    # 3. Create tensor processor for qwen3next
    class Qwen3NextTensorProcessor(TensorProcessor):
        HF_EXPERT_RENAME_PATTERN = re.compile(r"mlp\.experts\.\d+\.")
        HF_MOE_W13_PATTERN = re.compile(
            r"model\.layers\.(?P<bid>\d+)\.mlp\.experts\.gate_up_proj"
        )
        GGUF_MOE_WEIGHTS_PATTERN = re.compile(
            r"(?P<name>.*\.ffn_(?P<w>gate|down|up)_exps)\.weight$"
        )

        def __init__(self, config=None):
            super().__init__(config=config)

        def preprocess_name(self, hf_name: str) -> str:
            # dt_bias -> dt_proj so gguf name_map finds the SSM_DT mapping
            hf_name = hf_name.replace(".dt_bias", ".dt_proj")
            return re.sub(self.HF_EXPERT_RENAME_PATTERN, "mlp.experts.", hf_name)

        def perform_fallback_tensor_mapping(
            self, gguf_to_hf_name_map, suffix, qual_name, hf_name
        ):
            # Map merged MoE weights (w1 gate + w3 up) separately
            if m := re.fullmatch(self.HF_MOE_W13_PATTERN, hf_name):
                full_hf_name = qual_name + hf_name
                gguf_to_hf_name_map[
                    f"blk.{m['bid']}.ffn_gate_exps{suffix}"
                ] = full_hf_name
                gguf_to_hf_name_map[
                    f"blk.{m['bid']}.ffn_up_exps{suffix}"
                ] = full_hf_name

        def process(self, weights, name, **kwargs):
            if m := re.fullmatch(self.GGUF_MOE_WEIGHTS_PATTERN, name):
                tensor_key_mapping = kwargs.get("tensor_key_mapping")
                parsed_parameters = kwargs.get("parsed_parameters")
                if tensor_key_mapping:
                    # The GGUF file stores separate gate/up expert tensors but
                    # the gguf name_map creates a merged key (ffn_gate_up_exps).
                    # Map separate GGUF names to the merged mapping key.
                    gguf_key = m["name"]
                    if gguf_key not in tensor_key_mapping and m["w"] in (
                        "gate",
                        "up",
                    ):
                        bid = re.search(r"blk\.(\d+)\.", name).group(1)
                        gguf_key = f"blk.{bid}.ffn_gate_up_exps"
                    if gguf_key not in tensor_key_mapping:
                        return GGUFTensor(weights, None, {})
                    self._set_moe_expert_tensor(
                        weights,
                        parsed_parameters,
                        tensor_key_mapping[gguf_key],
                        m["w"],
                    )
                    return GGUFTensor(weights, None, {})
            if "ssm_conv1d" in name:
                # Conv1d weight must be (out_channels, 1, kernel_size)
                if weights.ndim == 2:
                    weights = np.expand_dims(weights, axis=1)
            if "ssm_a" in name:
                # Reverse the exponential: llama.cpp stores exp(A), HF expects log(-A)
                weights = np.log(-weights)
            if "ffn_gate_inp_shexp" in name:
                # shared_expert_gate must be (1, hidden_size), quantized is (hidden_size,)
                if weights.ndim == 1:
                    weights = np.expand_dims(weights, axis=0)
            return GGUFTensor(weights, name, {})

        def _set_moe_expert_tensor(self, weights, parsed_parameters, hf_name, w):
            torch_weights = torch.from_numpy(np.copy(weights))
            if w == "down":
                parsed_parameters["tensors"][hf_name] = torch_weights
            else:
                shape = list(weights.shape)
                shard_dim = 1
                shard_size = shape[shard_dim]
                shape[shard_dim] = shard_size * 2
                if hf_name not in parsed_parameters["tensors"]:
                    parsed_parameters["tensors"][hf_name] = torch.zeros(
                        shape, dtype=torch_weights.dtype
                    )
                out = parsed_parameters["tensors"][hf_name]
                if w == "gate":
                    out = out.narrow(shard_dim, 0, shard_size)
                else:  # w == "up"
                    out = out.narrow(shard_dim, shard_size, shard_size)
                out.copy_(torch_weights)

    TENSOR_PROCESSORS["qwen3next"] = Qwen3NextTensorProcessor

    # 3b. Register qwen3next tokenizer converter (same as qwen3/qwen2)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3next" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3next"] = GGUFQwen2Converter

    # 4. Patch load_gguf_checkpoint to handle qwen3next -> qwen3_next
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen3next":
            result["config"]["model_type"] = "qwen3_next"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Also patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 5. Patch get_gguf_hf_weights_map to handle qwen3_next -> qwen3next
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen3_next":
            model_type = "qwen3next"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


# Apply the monkey-patch at import time
_patch_transformers_qwen3next_gguf()


class ModelVariant(StrEnum):
    """Available Qwen 3 Next Abliterated GGUF model variants for causal language modeling."""

    QWEN_3_NEXT_80B_A3B_INSTRUCT_ABLITERATED_GGUF = "80B_A3B_Instruct_abliterated_GGUF"


class ModelLoader(ForgeModel):
    """Qwen 3 Next Abliterated GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_NEXT_80B_A3B_INSTRUCT_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-Next-80B-A3B-Instruct-abliterated-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_NEXT_80B_A3B_INSTRUCT_ABLITERATED_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_NEXT_80B_A3B_INSTRUCT_ABLITERATED_GGUF: "Qwen3-Next-80B-A3B-Instruct-abliterated.i1-Q4_K_M.gguf",
    }

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3 Next Abliterated GGUF",
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
        tokenizer_kwargs["gguf_file"] = self._gguf_file

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
        model_kwargs["gguf_file"] = self._gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._gguf_file
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

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
