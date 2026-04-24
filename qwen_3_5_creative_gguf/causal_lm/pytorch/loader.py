# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 Creative GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel


def _patch_transformers_qwen35moe_gguf():
    """Monkey-patch transformers to add qwen35moe GGUF architecture support.

    Transformers 5.x has Qwen3_5MoeForCausalLM but lacks GGUF loading support
    for the qwen35moe architecture. We bridge the gap by registering qwen35moe
    config/tensor mappings and converting the model_type to qwen3_5_moe_text.

    Also fixes the tensor_key_mapping for split MoE expert tensors: the gguf-py
    name map uses combined ffn_gate_up_exps but real GGUF files may store
    separate ffn_gate_exps and ffn_up_exps. Additionally, process() strips
    .weight from tensor names before lookup, so map keys must be without .weight.
    """
    import re

    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )

    # Fix get_gguf_hf_weights_map for qwen35moe split/combined expert tensors.
    # This must be applied once, even if architecture was already registered by
    # another loader, because process() looks up keys without .weight suffix but
    # the map is built with .weight suffix, and combined gate_up_exps must be
    # split into separate ffn_gate_exps and ffn_up_exps entries.
    if not getattr(gguf_utils, "_qwen35moe_weights_map_patched", False):
        _orig_get_map = gguf_utils.get_gguf_hf_weights_map

        def patched_get_gguf_hf_weights_map(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            _model_type = (
                hf_model.config.model_type if model_type is None else model_type
            )
            if _model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
                _model_type = "qwen35moe"
            result = _orig_get_map(
                hf_model, processor, _model_type, num_layers, qual_name
            )
            if _model_type == "qwen35moe":
                # process() strips .weight from GGUF tensor names before lookup,
                # so we need keys without .weight for MoE expert tensors.
                # Also, GGUF files store split gate/up (ffn_gate_exps,
                # ffn_up_exps) but the name map only has combined gate_up_exps.
                new_entries = {}
                for key, hf_name in list(result.items()):
                    # Match combined gate_up_exps with or without .weight suffix
                    m = re.fullmatch(r"(blk\.\d+)\.ffn_gate_up_exps(\.weight)?", key)
                    if m:
                        bid = m.group(1)
                        new_entries[f"{bid}.ffn_gate_exps"] = hf_name
                        new_entries[f"{bid}.ffn_up_exps"] = hf_name
                    # Match down_exps with or without .weight suffix
                    m2 = re.fullmatch(r"(blk\.\d+\.ffn_down_exps)(\.weight)?", key)
                    if m2:
                        new_entries[m2.group(1)] = hf_name
                result.update(new_entries)
            return result

        gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map
        gguf_utils._qwen35moe_weights_map_patched = True

    # Also patch Qwen2MoeTensorProcessor.process directly so split expert tensors
    # (ffn_gate_exps, ffn_up_exps) fall back to combined form (ffn_gate_up_exps.weight)
    # when the tensor_key_mapping doesn't have the split key. This is more robust than
    # relying on get_gguf_hf_weights_map being called with the right arguments.
    from transformers.modeling_gguf_pytorch_utils import (
        Qwen2MoeTensorProcessor,
        GGUFTensor,
    )
    import numpy as _np_patch

    if not getattr(Qwen2MoeTensorProcessor, "_split_expert_patched", False):
        _orig_proc_process = Qwen2MoeTensorProcessor.process

        def _patched_proc_process(self, weights, name, **kwargs):
            if m := re.fullmatch(self.GGUF_MOE_WEIGHTS_PATTERN, name):
                tensor_key_mapping = kwargs.get("tensor_key_mapping")
                parsed_parameters = kwargs.get("parsed_parameters")
                if tensor_key_mapping:
                    key = m["name"]
                    w = m["w"]
                    if key not in tensor_key_mapping:
                        if w in ("gate", "up"):
                            # GGUF stores split tensors; map has combined gate_up_exps.
                            # Try combined form without .weight first (direct param),
                            # then with .weight (nn.Linear param).
                            combined = re.sub(
                                r"ffn_(gate|up)_exps$", "ffn_gate_up_exps", key
                            )
                            if combined in tensor_key_mapping:
                                key = combined
                            elif combined + ".weight" in tensor_key_mapping:
                                key = combined + ".weight"
                        else:
                            if key + ".weight" in tensor_key_mapping:
                                key = key + ".weight"
                    if key in tensor_key_mapping:
                        self._set_moe_expert_tensor(
                            weights, parsed_parameters, tensor_key_mapping[key], w
                        )
                        return GGUFTensor(weights, None, {})
            if "ffn_gate_inp_shexp" in name:
                weights = _np_patch.expand_dims(weights, axis=0)
            return GGUFTensor(weights, name, {})

        Qwen2MoeTensorProcessor.process = _patched_proc_process
        Qwen2MoeTensorProcessor._split_expert_patched = True

    if "qwen35moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Architecture already registered by another loader

    # 1. Register qwen35moe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

    # 2. Add config mapping for qwen35moe (based on qwen3_moe + Qwen3.5 fields)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
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
    }

    # 3. Reuse qwen3moe tensor processor for qwen35moe
    if "qwen3moe" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["qwen35moe"] = TENSOR_PROCESSORS["qwen3moe"]

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

    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.models.auto.tokenization_auto as tok_auto

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


# Apply the monkey-patch at import time
_patch_transformers_qwen35moe_gguf()
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
    """Available Qwen 3.5 Creative GGUF model variants for causal language modeling."""

    QWEN_3_5_CREATIVE_26B_A3B_REAP_I1 = "26B_A3B_REAP_i1"
    QWEN_3_5_CREATIVE_26B_A3B_REAP = "26B_A3B_REAP"


class ModelLoader(ForgeModel):
    """Qwen 3.5 Creative GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP_I1: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-Creative-26B-A3B-REAP-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-Creative-26B-A3B-REAP-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP_I1

    _GGUF_FILES = {
        ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP_I1: "Qwen3.5-Creative-26B-A3B-REAP.i1-Q4_K_M.gguf",
        ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP: "Qwen3.5-Creative-26B-A3B-REAP.Q4_K_M.gguf",
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
            model="Qwen 3.5 Creative GGUF",
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._GGUF_FILES[self._variant]
            )
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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

    def _get_text_config(self):
        """Get the text config, handling both nested (MoE) and flat config structures."""
        if hasattr(self.config, "text_config"):
            return self.config.text_config
        return self.config

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        text_config = self._get_text_config()
        assert (
            text_config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
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
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
