# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Crow 9B HERETIC GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import numpy as np
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
    TENSOR_PROCESSORS,
    GGUFTensor,
    TensorProcessor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _find_real_load_gguf_checkpoint():
    """Walk the patch chain to find the true transformers load_gguf_checkpoint.

    Other loaders installed before this one may have replaced
    _gguf_utils.load_gguf_checkpoint with a broken wrapper that doesn't
    accept the model_to_load kwarg added in transformers 5.2.0.  Broken
    wrappers store their captured original as _orig_load_gguf_checkpoint in
    the wrapper function's module globals; walk that chain until we reach the
    function whose __name__ and __module__ match the real transformers impl.
    """
    fn = _gguf_utils.load_gguf_checkpoint
    seen: set = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        if (
            getattr(fn, "__name__", "") == "load_gguf_checkpoint"
            and getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils"
        ):
            return fn
        advanced = False
        # Walk via closure cells (for wrappers that close over local variables)
        if fn.__closure__:
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val) and "load_gguf_checkpoint" in getattr(val, "__name__", ""):
                        fn = val
                        advanced = True
                        break
                except ValueError:
                    pass
        if not advanced:
            # Walk via module globals (pattern: from ... import load_gguf_checkpoint as _orig)
            for var in ("_orig_load_gguf_checkpoint", "_orig", "_real_orig"):
                nxt = getattr(fn, "__globals__", {}).get(var)
                if callable(nxt) and nxt is not fn:
                    fn = nxt
                    advanced = True
                    break
        if not advanced:
            break
    return fn


_orig_load_gguf_checkpoint = _find_real_load_gguf_checkpoint()

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


def _patch_qwen35_support():
    """Register qwen35 architecture and qwen3_5_text tokenizer as aliases for qwen3.

    Qwen 3.5 GGUF files declare architecture 'qwen35' and tokenizer class
    'qwen3_5_text', which transformers 5.x does not yet recognise natively.
    """
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen35 support and fix model_type.

    Uses *args/**kwargs to remain compatible with transformers 5.2.0 which
    added the model_to_load keyword argument.  Maps the GGUF 'qwen35'
    architecture to the transformers 'qwen3_5_text' model type and derives
    layer_types from full_attention_interval (defaulting to 4).
    """
    _patch_qwen35_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
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


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to map qwen3_5_text -> qwen35 for gguf-py lookup."""
    if model_type is None:
        model_type = hf_model.config.model_type
    if model_type in ("qwen3_5_text", "qwen3_5"):
        model_type = "qwen35"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


class _Qwen35TensorProcessor(TensorProcessor):
    """Process Qwen3.5 GGUF tensors for HuggingFace model loading."""

    def preprocess_name(self, hf_name: str) -> str:
        return hf_name.replace(".dt_bias", ".dt_proj")

    def process(self, weights, name, **kwargs):
        if "ssm_conv1d.weight" in name:
            if weights.ndim == 2:
                weights = np.expand_dims(weights, axis=1)
        if "ssm_a" in name and "ssm_alpha" not in name:
            weights = np.log(-weights)
        return GGUFTensor(weights, name, {})


def _install_patches():
    """Install load_gguf_checkpoint and get_gguf_hf_weights_map patches.

    Called at module level AND inside load_model() to win against any
    broken overrides installed by later-collected loader modules.
    """
    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
    TENSOR_PROCESSORS["qwen35"] = _Qwen35TensorProcessor


_patch_qwen35_support()
_install_patches()


class ModelVariant(StrEnum):
    """Available Crow 9B HERETIC GGUF model variants for causal language modeling."""

    CROW_9B_HERETIC_GGUF = "9B_HERETIC_GGUF"


class ModelLoader(ForgeModel):
    """Crow 9B HERETIC GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.CROW_9B_HERETIC_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Crow-9B-HERETIC-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CROW_9B_HERETIC_GGUF

    GGUF_FILE = "Crow-9B-HERETIC.Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="Crow 9B HERETIC GGUF",
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

        # Re-apply patches here because other loaders imported during pytest
        # collection may have overwritten load_gguf_checkpoint with a version
        # that lacks the model_to_load kwarg added in transformers 5.2.0.
        _install_patches()

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

        # Qwen3_5DynamicCache is not a pytree-registered type, so the test
        # infrastructure's tree_map comparison fails if past_key_values is returned.
        # Disable caching for a single forward pass — logits are unaffected.
        inputs["use_cache"] = False

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

            # Qwen3.5 uses self_attn for full-attention layers only;
            # linear-attention layers (Qwen3_5GatedDeltaNet) use linear_attn.
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.q_proj.bias] = ("model",)
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.bias] = ("model",)
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.bias] = ("model",)
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
