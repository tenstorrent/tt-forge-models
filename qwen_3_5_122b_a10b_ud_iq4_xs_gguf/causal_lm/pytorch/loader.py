# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5-122B-A10B UD-IQ4_XS GGUF (DanyDA split) model loader implementation for causal language modeling.
"""
import importlib.metadata

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


class ModelVariant(StrEnum):
    """Available Qwen3.5-122B-A10B UD-IQ4_XS GGUF model variants for causal language modeling."""

    QWEN_3_5_122B_A10B_UD_IQ4_XS_GGUF = "122B_A10B_UD_IQ4_XS_GGUF"


class ModelLoader(ForgeModel):
    """Qwen3.5-122B-A10B UD-IQ4_XS GGUF (DanyDA split) model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_122B_A10B_UD_IQ4_XS_GGUF: LLMModelConfig(
            pretrained_model_name="DanyDA/unsloth_Qwen3.5-122B-A10B-GGUF_UD-IQ4_XS-GGUF-SPLIT",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_122B_A10B_UD_IQ4_XS_GGUF

    GGUF_FILE = "unsloth_Qwen3.5-122B-A10B-GGUF_UD-IQ4_XS-00001-of-00013.gguf"

    sample_text = "Give me a short introduction to large language models."

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager.

        transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
        is installed later, the mapping is stale and version detection falls back to
        gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
        """
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    @staticmethod
    def _patch_qwen35moe_tensor_mapping():
        """Patch GGUF loading to correctly handle qwen35moe architecture for 122B.

        Fixes two issues:
        1. The qwen3_moe GGUF config mapping lacks expert_feed_forward_length, so
           moe_intermediate_size defaults to 512 instead of the correct 1024.
        2. The 122B GGUF uses separate ffn_gate_exps/ffn_up_exps (qwen3moe naming)
           instead of merged ffn_gate_up_exps (qwen35moe naming). Forcing qwen3moe
           name_map triggers the fallback that correctly maps these split tensors.
        """
        import transformers.integrations.ggml as ggml_module
        import transformers.modeling_gguf_pytorch_utils as gguf_utils

        # Fix moe_intermediate_size: the qwen3_moe GGUF config mapping is missing
        # expert_feed_forward_length, so the model initializes with the wrong expert
        # intermediate size (512 default instead of 1024 from GGUF metadata).
        qwen3_moe_map = ggml_module.GGUF_CONFIG_MAPPING.get("qwen3_moe", {})
        if "expert_feed_forward_length" not in qwen3_moe_map:
            qwen3_moe_map["expert_feed_forward_length"] = "moe_intermediate_size"

        _qwen35moe_variants = frozenset(
            {"qwen3_5_moe_text", "qwen3_5_moe", "qwen35moe"}
        )

        # Ensure qwen35moe arch has a processor (GGUF header declares qwen35moe)
        if "qwen35moe" not in gguf_utils.TENSOR_PROCESSORS:
            qwen3moe_processor = gguf_utils.TENSOR_PROCESSORS.get("qwen3moe")
            if qwen3moe_processor is not None:
                gguf_utils.TENSOR_PROCESSORS["qwen35moe"] = qwen3moe_processor

        # Ensure qwen35moe is in the supported architectures list
        if "qwen35moe" not in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
            gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

        if getattr(gguf_utils.get_gguf_hf_weights_map, "_qwen122b_patched", False):
            return

        _orig = gguf_utils.get_gguf_hf_weights_map

        def _patched(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            if model_type is None:
                model_type = getattr(
                    getattr(hf_model, "config", None), "model_type", None
                )
            if model_type in _qwen35moe_variants:
                model_type = "qwen3moe"
            return _orig(
                hf_model,
                processor,
                model_type=model_type,
                num_layers=num_layers,
                qual_name=qual_name,
            )

        _patched._qwen122b_patched = True
        gguf_utils.get_gguf_hf_weights_map = _patched

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
            model="Qwen3.5-122B-A10B UD-IQ4_XS GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
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
        self._fix_gguf_version_detection()
        self._patch_qwen35moe_tensor_mapping()
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
        self._fix_gguf_version_detection()
        self._patch_qwen35moe_tensor_mapping()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
