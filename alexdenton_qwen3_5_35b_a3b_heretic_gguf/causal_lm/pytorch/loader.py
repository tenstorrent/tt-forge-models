# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
alexdenton Qwen3.5 35B A3B Heretic GGUF model loader implementation for causal language modeling.
"""
import re
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


def _get_gguf_arch_and_interval(gguf_path):
    """Read architecture and full_attention_interval from GGUF header."""
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(gguf_path))
        arch = ""
        interval = 4
        for key, field in reader.fields.items():
            if key == "general.architecture":
                parts = [field.parts[i] for i in field.data]
                if parts:
                    arch = parts[0].tobytes().decode("utf-8")
            elif "full_attention_interval" in key:
                parts = [field.parts[i] for i in field.data]
                if parts:
                    val = parts[0]
                    interval = int(val.item()) if hasattr(val, "item") else int(val)
        return arch, interval
    except Exception:
        return "", 4


# Match combined gate_up entry (with or without .weight suffix)
_COMBINED_EXPS = re.compile(r"blk\.(\d+)\.ffn_gate_up_exps(?:\.weight)?$")
# Match individual exp entries that have .weight (need bare-key alias)
_EXPS_WITH_WEIGHT = re.compile(r"(blk\.\d+\.ffn_(?:gate|up|down)_exps)\.weight$")


def _augment_qwen35moe_weights_map(result):
    """Add separate gate/up entries and bare-key (no .weight) aliases for qwen35moe.

    When the HF model uses gate_up_proj (combined) but the GGUF file has separate
    ffn_gate_exps / ffn_up_exps tensors, Qwen2MoeTensorProcessor.process() needs
    blk.N.ffn_gate_exps and blk.N.ffn_up_exps in the map pointing to the combined
    HF parameter. Also handles .weight suffix mismatch between map keys and lookup.
    """
    extra = {}
    for key, val in result.items():
        # When map has combined gate_up (with or without .weight), add separate entries
        m = _COMBINED_EXPS.match(key)
        if m:
            bid = m.group(1)
            extra.setdefault(f"blk.{bid}.ffn_gate_exps", val)
            extra.setdefault(f"blk.{bid}.ffn_up_exps", val)
        # When map has .weight-suffixed keys, add bare-key aliases for process() lookup
        m2 = _EXPS_WITH_WEIGHT.match(key)
        if m2:
            extra.setdefault(m2.group(1), val)
    result.update(extra)
    return result


def _patch_transformers_qwen35moe_gguf():
    """Patch transformers to support qwen35moe GGUF loading for Qwen3.5 MoE models.

    Transformers string-replaces 'qwen35moe' -> 'qwen3_moe' (substring match on 'qwen3moe'),
    which prevents correct Qwen3.5 MoE loading. This patch:
    1. Fixes the config model_type and layer_types after the first (config-only) GGUF load
    2. Fixes the MoE expert tensor key mapping so Qwen2MoeTensorProcessor.process() works
       (process() looks up keys without .weight suffix, but the map uses .weight suffixes)

    The get_gguf_hf_weights_map fix is re-applied at call time inside the patched
    load_gguf_checkpoint to survive later loaders overwriting the module-level reference.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )

    if getattr(gguf_utils, "_alexdenton_qwen35moe_patched", False):
        return
    gguf_utils._alexdenton_qwen35moe_patched = True

    # Register qwen35moe as supported (may already be done by other loaders)
    if "qwen35moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

    if "qwen35moe" not in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
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

    if "qwen3moe" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS.setdefault("qwen35moe", TENSOR_PROCESSORS["qwen3moe"])

    try:
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

        if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS.setdefault(
                "qwen35moe", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
            )
            GGUF_TO_FAST_CONVERTERS.setdefault(
                "qwen3_5_moe_text", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
            )
    except Exception:
        pass

    _orig_load = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None
    ):
        if return_tensors:
            # Re-apply the weights-map fix at call time so it survives later patches.
            # Wrap whatever get_gguf_hf_weights_map is currently installed.
            _current_get_map = gguf_utils.get_gguf_hf_weights_map

            def _augmented_get_map(
                hf_model, processor, model_type=None, num_layers=None, qual_name=""
            ):
                if model_type is None:
                    model_type = hf_model.config.model_type
                # Ensure qwen3_5_moe_text routes to qwen35moe tensor processor
                if model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
                    model_type = "qwen35moe"
                result = _current_get_map(
                    hf_model, processor, model_type, num_layers, qual_name
                )
                _augment_qwen35moe_weights_map(result)
                return result

            gguf_utils.get_gguf_hf_weights_map = _augmented_get_map
            try:
                return _orig_load(gguf_checkpoint_path, return_tensors, model_to_load)
            finally:
                gguf_utils.get_gguf_hf_weights_map = _current_get_map

        result = _orig_load(gguf_checkpoint_path, return_tensors, model_to_load)

        # Fix config on the first (config-only) call so the right model class is used
        arch, interval = _get_gguf_arch_and_interval(gguf_checkpoint_path)
        if arch == "qwen35moe" and result.get("config", {}).get("model_type") in (
            "qwen3_moe",
            "qwen35moe",
        ):
            config = result["config"]
            config["model_type"] = "qwen3_5_moe_text"
            num_layers = config.get("num_hidden_layers", 40)
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(num_layers)
            ]
            config["layer_types"] = layer_types
            config.pop("full_attention_interval", None)

        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    import transformers.models.auto.tokenization_auto as _tok_auto
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_utils as _modeling_utils

    for _mod in (_tok_auto, _config_utils, _modeling_utils):
        if hasattr(_mod, "load_gguf_checkpoint"):
            _mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_transformers_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available alexdenton Qwen3.5 35B A3B Heretic GGUF model variants for causal language modeling."""

    QWEN_3_5_35B_A3B_HERETIC_GGUF = "35B_A3B_HERETIC_GGUF"


class ModelLoader(ForgeModel):
    """alexdenton Qwen3.5 35B A3B Heretic GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_35B_A3B_HERETIC_GGUF: LLMModelConfig(
            pretrained_model_name="alexdenton/Qwen3.5-35B-A3B-heretic-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_35B_A3B_HERETIC_GGUF

    GGUF_FILE = "Qwen3.5-35B-A3B-heretic-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="alexdenton Qwen3.5 35B A3B Heretic GGUF",
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
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
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
