# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5-text-9B GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata as _importlib_metadata
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.utils.import_utils as _import_utils
from packaging import version as _packaging_version
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patched_is_gguf_available(min_version=_import_utils.GGUF_MIN_VERSION):
    """Check gguf availability with a fresh metadata lookup.

    transformers caches importlib.metadata.packages_distributions() at module
    import time.  When gguf is installed dynamically (via requirements.txt and
    RequirementsManager), the stale cache causes gguf.__version__ to fall back
    to 'N/A', raising packaging.version.InvalidVersion.  Bypassing the stale
    mapping by calling importlib.metadata.version() directly fixes the check.
    """
    try:
        ver = _importlib_metadata.version("gguf")
        return _packaging_version.parse(ver) >= _packaging_version.parse(min_version)
    except (
        _importlib_metadata.PackageNotFoundError,
        _packaging_version.InvalidVersion,
    ):
        return False


_gguf_utils.is_gguf_available = _patched_is_gguf_available


def _patch_qwen35_text_support():
    """Upgrade the qwen35 GGUF mapping so Qwen3.5-text loads as qwen3_5_text.

    Other loaders (bartowski_coniccat, etc.) register qwen35 as an alias for
    qwen3 and convert the model_type to "qwen3".  Qwen3.5-text has a hybrid
    attention architecture (full_attention_interval) that Qwen3ForCausalLM
    does not support; it needs Qwen3_5ForCausalLM via model_type "qwen3_5_text".

    We fix this in three steps:
    1. Ensure full_attention_interval and head_dim are captured from the GGUF.
    2. Wrap load_gguf_checkpoint to upgrade "qwen3" configs that carry
       full_attention_interval to "qwen3_5_text" and synthesise layer_types.
    3. Wrap get_gguf_hf_weights_map to map "qwen3_5_text" → "qwen35" so the
       tensor name lookup uses the correct GGUF architecture.
    """
    # Step 1: upgrade the qwen35 config field mapping (set by bartowski et al.)
    cfg_map = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING.get("config", {})
    qwen35_map = cfg_map.get("qwen35")
    if qwen35_map is not None and "full_attention_interval" not in qwen35_map:
        # Avoid modifying the shared qwen3 mapping object in-place.
        new_map = dict(qwen35_map)
        new_map["full_attention_interval"] = "full_attention_interval"
        new_map.setdefault("attention.key_length", "head_dim")
        cfg_map["qwen35"] = new_map

    # Step 2: wrap load_gguf_checkpoint
    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        config = result.get("config", {})
        if "full_attention_interval" in config and config.get("model_type") in (
            "qwen3",
            "qwen35",
        ):
            interval = config.pop("full_attention_interval")
            num_layers = config.get("num_hidden_layers", 32)
            config["layer_types"] = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(num_layers)
            ]
            config["model_type"] = "qwen3_5_text"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load

    import transformers.configuration_utils as _config_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer

    for _mod in (_config_utils, _auto_tokenizer):
        if hasattr(_mod, "load_gguf_checkpoint"):
            _mod.load_gguf_checkpoint = _patched_load

    # Step 3: wrap get_gguf_hf_weights_map
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hf_model is not None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_text", "qwen3_5"):
            model_type = "qwen35"
        return _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_map


_patch_qwen35_text_support()


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
    """Available Qwen3.5-text-9B GGUF model variants for causal language modeling."""

    QWEN_3_5_TEXT_9B_Q4_K_M_GGUF = "text_9B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Qwen3.5-text-9B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_TEXT_9B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="techwithsergiu/Qwen3.5-text-9B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_TEXT_9B_Q4_K_M_GGUF

    GGUF_FILE = "Qwen3.5-text-9B-Q4_K_M.gguf"

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
            model="Qwen3.5-text-9B GGUF",
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
