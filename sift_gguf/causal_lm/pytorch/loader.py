# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sift GGUF model loader implementation for causal language modeling.

Sift is a Qwen3.5 SSM hybrid model with full attention every 4 layers.
The GGUF architecture is 'qwen35' which must be loaded as 'qwen3_5_text'
(Qwen3_5ForCausalLM) rather than 'qwen3' (Qwen3ForCausalLM) to correctly
handle the interleaved linear-attention and full-attention layers.
"""
import torch
from gguf import GGUFReader
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
    TENSOR_PROCESSORS,
    MambaTensorProcessor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

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

_QWEN35_SSM_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.key_length": "head_dim",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
    "ssm.conv_kernel": "linear_conv_kernel_dim",
    "ssm.state_size": "linear_key_head_dim",
    "ssm.group_count": "linear_num_key_heads",
    "ssm.time_step_rank": "linear_num_value_heads",
    "ssm.inner_size": None,
    "full_attention_interval": None,
}


def _patch_qwen35_ssm_support():
    """Register qwen35 SSM hybrid architecture with proper config and weight mapping.

    Qwen3.5 SSM hybrid uses 'qwen35' in GGUF but maps to Qwen3_5ForCausalLM
    in transformers, not Qwen3ForCausalLM. The hybrid has linear_attention
    (SSM) layers and full_attention layers interleaved.
    """
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    # Copy all section mappings from qwen3 (tokenizer, etc.) first
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )

    # Override config section with SSM-aware mapping
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "qwen35"
    ] = _QWEN35_SSM_CONFIG_MAPPING

    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )

    # Register MambaTensorProcessor for qwen35 so ssm_conv1d.weight gets the
    # missing channel dimension added ([N,4] -> [N,1,4]) and ssm_a values are
    # log-transformed to A_log.
    TENSOR_PROCESSORS.setdefault("qwen35", MambaTensorProcessor)


def _is_qwen35_gguf(gguf_path):
    """Return True if the GGUF file's general.architecture is 'qwen35'."""
    try:
        reader = GGUFReader(gguf_path)
        arch_field = reader.fields.get("general.architecture")
        if arch_field is None:
            return False
        # STRING fields: parts[data[0]] is a uint8 memmap of the UTF-8 bytes
        arch = arch_field.parts[arch_field.data[0]].tobytes().decode("utf-8")
        return arch == "qwen35"
    except Exception:
        return False


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    """Load GGUF checkpoint with Qwen3.5 SSM hybrid support.

    After the standard load, converts the qwen35 config to qwen3_5_text and
    generates layer_types from full_attention_interval stored in the GGUF.

    Reads the GGUF architecture directly to detect qwen35 files even when
    another loader's patch has already transformed model_type to 'qwen3'.
    """
    _patch_qwen35_ssm_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
    )
    config = result.get("config", {})

    # Use GGUF file architecture as ground truth: another loader imported before
    # us may have already mapped model_type from "qwen35" to "qwen3".
    is_qwen35 = config.get("model_type") == "qwen35" or _is_qwen35_gguf(gguf_path)
    if not is_qwen35:
        return result

    config["model_type"] = "qwen3_5_text"
    config["architectures"] = ["Qwen3_5ForCausalLM"]

    # ssm.state_size maps to linear_key_head_dim; copy to linear_value_head_dim too
    if "linear_key_head_dim" in config and "linear_value_head_dim" not in config:
        config["linear_value_head_dim"] = config["linear_key_head_dim"]

    # Generate layer_types from full_attention_interval in the GGUF metadata
    try:
        reader = GGUFReader(gguf_path)
        full_attn_field = reader.fields.get("qwen35.full_attention_interval")
        if full_attn_field is not None:
            interval = int(full_attn_field.parts[full_attn_field.data[0]][0])
            num_layers = config.get("num_hidden_layers", 24)
            config["layer_types"] = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(num_layers)
            ]
    except Exception:
        pass

    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Map qwen3_5_text model type to qwen35 gguf-py architecture name.

    Also injects the ssm_dt.bias -> linear_attn.dt_bias mapping that gguf-py's
    QWEN35 tensor name map omits (it has 'ssm_dt' without the '.bias' suffix).
    """
    if model_type is None and hasattr(hf_model, "config"):
        model_type = hf_model.config.model_type
    if model_type in ("qwen3_5_text", "qwen3_5"):
        model_type = "qwen35"
    result = _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )
    if model_type == "qwen35" and qual_name == "":
        n = num_layers
        if n is None and hasattr(hf_model, "config"):
            n = hf_model.config.num_hidden_layers
        if n is not None:
            for i in range(n):
                gguf_key = f"blk.{i}.ssm_dt.bias"
                if gguf_key not in result:
                    result[gguf_key] = f"model.layers.{i}.linear_attn.dt_bias"
    return result


_patch_qwen35_ssm_support()


def _apply_qwen35_patches():
    _patch_qwen35_ssm_support()
    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


def _restore_patches(saved):
    _gguf_utils.load_gguf_checkpoint = saved["load_gguf_checkpoint"]
    _config_utils.load_gguf_checkpoint = saved["config_load_gguf_checkpoint"]
    _auto_tokenizer.load_gguf_checkpoint = saved["auto_tokenizer_load_gguf_checkpoint"]
    _tok_utils.load_gguf_checkpoint = saved["tok_utils_load_gguf_checkpoint"]
    _gguf_utils.get_gguf_hf_weights_map = saved["get_gguf_hf_weights_map"]


def _save_current_patches():
    return {
        "load_gguf_checkpoint": _gguf_utils.load_gguf_checkpoint,
        "config_load_gguf_checkpoint": _config_utils.load_gguf_checkpoint,
        "auto_tokenizer_load_gguf_checkpoint": _auto_tokenizer.load_gguf_checkpoint,
        "tok_utils_load_gguf_checkpoint": _tok_utils.load_gguf_checkpoint,
        "get_gguf_hf_weights_map": _gguf_utils.get_gguf_hf_weights_map,
    }


class ModelVariant(StrEnum):
    """Available Sift GGUF model variants for causal language modeling."""

    SIFT_GGUF = "GGUF"


class ModelLoader(ForgeModel):
    """Sift GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SIFT_GGUF: LLMModelConfig(
            pretrained_model_name="Sid77449/sift",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SIFT_GGUF

    GGUF_FILE = "model.gguf"

    sample_text = "Clean this error: Traceback (most recent call last): File /tmp/foo.py line 10 ValueError invalid literal for int"

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
            model="Sift GGUF",
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

        saved = _save_current_patches()
        try:
            _apply_qwen35_patches()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name, **tokenizer_kwargs
            )
        finally:
            _restore_patches(saved)

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

        saved = _save_current_patches()
        try:
            _apply_qwen35_patches()

            if self.num_layers is not None:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
                config.num_hidden_layers = self.num_layers
                if hasattr(config, "layer_types") and config.layer_types:
                    config.layer_types = config.layer_types[: self.num_layers]
                model_kwargs["config"] = config

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _restore_patches(saved)

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
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            elif hasattr(layer, "linear_attn"):
                shard_specs[layer.linear_attn.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        saved = _save_current_patches()
        try:
            _apply_qwen35_patches()
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        finally:
            _restore_patches(saved)
        return self.config
