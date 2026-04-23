# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive GGUF model loader implementation for causal language modeling.
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
    """Monkey-patch transformers to support qwen35moe GGUF with separate gate/up expert tensors.

    Qwen3.5 MoE GGUF files store expert weights as separate ffn_gate_exps and
    ffn_up_exps tensors, but the gguf-py qwen35moe name map only maps gate_up_proj
    to ffn_gate_up_exps (combined). We fix this by adding the separate gate/up
    entries to the tensor key mapping so the Qwen2MoeTensorProcessor can combine
    them into gate_up_proj during loading.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen35moe" in GGUF_SUPPORTED_ARCHITECTURES:
        # Already patched — still apply the get_gguf_hf_weights_map fix below
        pass
    else:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

        if "qwen3moe" in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
            base = GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3moe"]
            GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"] = dict(base)
            GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"][
                "full_attention_interval"
            ] = "full_attention_interval"

        if "qwen3moe" in TENSOR_PROCESSORS:
            TENSOR_PROCESSORS["qwen35moe"] = TENSOR_PROCESSORS["qwen3moe"]

        if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS.setdefault(
                "qwen35moe", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
            )
            GGUF_TO_FAST_CONVERTERS.setdefault(
                "qwen3_5_moe_text", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
            )

    # Patch get_gguf_hf_weights_map to:
    # 1. Convert qwen3_5_moe_text -> qwen35moe for gguf-py arch lookup
    # 2. Add ffn_gate_exps/ffn_up_exps -> gate_up_proj entries (GGUF stores
    #    gate and up separately but qwen35moe name map only has ffn_gate_up_exps)
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
            model_type = "qwen35moe"
        result = orig_get_map(hf_model, processor, model_type, num_layers, qual_name)
        if model_type == "qwen35moe":
            extra = {}
            for k, v in result.items():
                if "ffn_gate_up_exps" in k:
                    extra[k.replace("ffn_gate_up_exps", "ffn_gate_exps")] = v
                    extra[k.replace("ffn_gate_up_exps", "ffn_up_exps")] = v
            result.update(extra)
        return result

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map

    # Patch load_gguf_checkpoint to convert qwen35moe -> qwen3_5_moe_text and
    # generate layer_types from full_attention_interval
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

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.tokenization_utils_tokenizers as tok_utils

    for mod in (tok_auto, config_utils, modeling_utils, tok_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive GGUF model variants for causal language modeling."""

    MIRXA_3_5_35B_A3B_UNCENSORED_AGGRESSIVE_Q4_K_M_GGUF = (
        "3_5_35B_A3B_Uncensored_Aggressive_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MIRXA_3_5_35B_A3B_UNCENSORED_AGGRESSIVE_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mirxa2/Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIRXA_3_5_35B_A3B_UNCENSORED_AGGRESSIVE_Q4_K_M_GGUF

    GGUF_FILE = "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

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
            model="Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive GGUF",
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
        model_kwargs["ignore_mismatched_sizes"] = True

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
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
            enable_thinking=True,
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
