# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Qwen3.5-27B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _patch_qwen35_support():
    """Register qwen35 architecture and qwen3_5_text tokenizer as aliases.

    Qwen3.5-27B is a hybrid SSM+Attention model. The GGUF file declares
    architecture as 'qwen35', which maps to the HF 'qwen3_5_text' model type.
    """
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    qwen3_config = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].get("qwen3", {})
    qwen35_config = {
        **qwen3_config,
        # SSM/linear-attention specific mappings
        "ssm.conv_kernel": "linear_conv_kernel_dim",
        "ssm.group_count": "linear_num_key_heads",
        "ssm.time_step_rank": "linear_num_value_heads",
        "ssm.state_size": "linear_key_head_dim",
        "attention.key_length": "head_dim",
        # full_attention_interval is handled in post-processing (see _patched_load_gguf_checkpoint)
        "full_attention_interval": None,
    }
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "qwen35", qwen35_config
    )

    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to map qwen3_5_text -> qwen35 GGUF arch."""
    effective_model_type = model_type
    if effective_model_type is None and hasattr(hf_model, "config"):
        effective_model_type = hf_model.config.model_type
    if effective_model_type == "qwen3_5_text":
        model_type = "qwen35"
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen35 support and set correct model_type."""
    _patch_qwen35_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    config = result.get("config", {})
    if config.get("model_type") == "qwen35":
        config["model_type"] = "qwen3_5_text"

        # linear_value_head_dim equals linear_key_head_dim (same SSM state size)
        if "linear_key_head_dim" in config:
            config.setdefault("linear_value_head_dim", config["linear_key_head_dim"])

        # Compute layer_types from full_attention_interval in GGUF metadata
        gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path", "")
        if gguf_path:
            try:
                from gguf import GGUFReader

                reader = GGUFReader(gguf_path)
                if "qwen35.full_attention_interval" in reader.fields:
                    field = reader.fields["qwen35.full_attention_interval"]
                    interval = int(field.parts[field.data[0]])
                    num_layers = config.get("num_hidden_layers", 64)
                    layer_types = [
                        "full_attention"
                        if (i + 1) % interval == 0
                        else "linear_attention"
                        for i in range(num_layers)
                    ]
                    config["layer_types"] = layer_types

                # GGUF attention.head_count for qwen35 doesn't match the actual
                # q_proj tensor shape. Read the true num_attention_heads from the
                # first attn_q tensor found (only present on full-attention layers).
                head_dim = config.get("head_dim", 256)
                for tensor in reader.tensors:
                    if "attn_q.weight" in tensor.name:
                        q_out = max(tensor.shape)
                        if q_out % head_dim == 0:
                            config["num_attention_heads"] = q_out // head_dim
                        break
            except Exception:
                pass
    return result


_patch_qwen35_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
    """Available mradermacher Qwen3.5-27B GGUF model variants for causal language modeling."""

    MRADERMACHER_QWEN3_5_27B_GGUF = "27B_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher Qwen3.5-27B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MRADERMACHER_QWEN3_5_27B_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MRADERMACHER_QWEN3_5_27B_GGUF

    GGUF_FILE = "Qwen3.5-27B.Q4_K_M.gguf"

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
            model="mradermacher Qwen3.5-27B GGUF",
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

        # Always pre-load config explicitly so AutoModelForCausalLM resolves
        # Qwen3_5ForCausalLM (not Qwen3ForCausalLM from a hub config.json).
        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        inner_config = getattr(config, "text_config", config)
        if self.num_layers is not None:
            inner_config.num_hidden_layers = self.num_layers
            if (
                hasattr(inner_config, "layer_types")
                and inner_config.layer_types is not None
            ):
                inner_config.layer_types = inner_config.layer_types[: self.num_layers]
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
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

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
