# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bjivanovich Qwen3.5-4B-Vision GGUF model loader implementation for causal language modeling.
"""
import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils

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

# ---------------------------------------------------------------------------
# Qwen3.5 SSM/hybrid GGUF support
#
# The bjivanovich Qwen3.5-4B-Vision GGUF stores the architecture string as
# "qwen35" (an SSM/linear-attention hybrid with full_attention_interval=4).
# transformers 5.x does not map "qwen35" to Qwen3_5Config/ForCausalLM by
# default, and the 26+ other qwen35 loaders that patch load_gguf_checkpoint
# at module level remap model_type "qwen35" → "qwen3" (pure attention),
# which produces shape mismatches because Qwen3_5 uses q_proj *= 2 and a
# different head_dim.
#
# The context manager below:
#   1. Forcefully sets the GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35"]
#      entry to include "attention.key_length" → "head_dim" and
#      "full_attention_interval" → "full_attention_interval" so Qwen3_5Config
#      receives the right values.
#   2. Finds the REAL transformers load_gguf_checkpoint by walking the
#      patching chain, bypassing any outer qwen35→qwen3 remappers.
#   3. Patches all four binding sites with a wrapper that calls the real
#      function and remaps model_type "qwen35" → "qwen3_5".
#   4. Patches get_gguf_hf_weights_map so that Qwen3_5ForCausalLM (which has
#      config.model_type="qwen3_5") resolves to the "qwen35" gguf-py arch.
#   5. Restores everything on exit.
# ---------------------------------------------------------------------------


def _make_qwen35_tensor_processor():
    """Create a TensorProcessor subclass for qwen35 (Qwen3.5 SSM/hybrid) GGUF checkpoints."""
    from transformers.modeling_gguf_pytorch_utils import TensorProcessor, GGUFTensor
    import numpy as np

    class Qwen35TensorProcessor(TensorProcessor):
        def process(self, weights, name, **kwargs):
            # Reshape conv1d weight: GGUF [out, kernel] → PyTorch [out, 1, kernel]
            if "ssm_conv1d.weight" in name:
                weights = np.expand_dims(weights, axis=1)

            # Transform the A_log parameter:
            # GGUF stores raw A values (negative), model expects A_log = log(|A|)
            # Match only the bare "ssm_a" tensor, not "ssm_alpha" or other names.
            basename = name.rsplit(".", 1)[0].split(".")[-1]
            if basename == "ssm_a":
                weights = np.log(-weights)

            return GGUFTensor(weights, name, {})

    return Qwen35TensorProcessor


def _register_qwen35_vision_config():
    """Register qwen35 in GGUF_TO_TRANSFORMERS_MAPPING with SSM hybrid fields."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        TENSOR_PROCESSORS,
    )

    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    qwen35_config_map = {
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
        "full_attention_interval": "full_attention_interval",
        "vocab_size": "vocab_size",
    }
    # Forcefully set (not setdefault) so we override any prior qwen35→qwen3 alias
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35"] = qwen35_config_map

    # Register custom tensor processor for SSM/conv1d weight reshaping
    TENSOR_PROCESSORS["qwen35"] = _make_qwen35_tensor_processor()

    try:
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
        if "qwen3" in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["qwen35"] = GGUF_TO_FAST_CONVERTERS["qwen3"]
    except ImportError:
        pass


def _is_qwen35_ssm_gguf(gguf_path):
    """Return True if this GGUF file is a qwen35 SSM/hybrid model."""
    try:
        import struct
        import gguf as _gguf_py
        reader = _gguf_py.GGUFReader(gguf_path)
        arch_field = reader.fields.get("general.architecture")
        if arch_field is None:
            return False
        arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
        if arch != "qwen35":
            return False
        return reader.fields.get("qwen35.full_attention_interval") is not None
    except Exception:
        return False


def _get_qwen35_vision_config_overrides(gguf_path):
    """Extract SSM-specific config fields from qwen35 GGUF metadata."""
    import struct
    import gguf as _gguf_py
    reader = _gguf_py.GGUFReader(gguf_path)
    overrides = {}
    field = reader.fields.get("qwen35.full_attention_interval")
    if field:
        overrides["full_attention_interval"] = struct.unpack(
            "<I", bytes(field.parts[field.data[0]])
        )[0]
    field = reader.fields.get("qwen35.attention.key_length")
    if field:
        overrides["head_dim"] = struct.unpack(
            "<I", bytes(field.parts[field.data[0]])
        )[0]
    return overrides


@contextlib.contextmanager
def _qwen35_vision_gguf_context():
    """Context manager that routes qwen35 GGUF loading through Qwen3_5ForCausalLM."""
    _register_qwen35_vision_config()

    # Capture the current (possibly already-patched) chain.
    _chain_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load(gguf_path, return_tensors=False, **kwargs):
        # Ensure qwen35 is registered (in case another loader reset the mapping).
        _register_qwen35_vision_config()
        result = _chain_load(gguf_path, return_tensors=return_tensors, **kwargs)
        config = result.get("config", {})
        # The chain may have already remapped qwen35 → qwen3 (tvall43-style
        # patchers do this).  Re-read the GGUF to detect the SSM architecture
        # and override with qwen3_5_text (Qwen3_5TextConfig / Qwen3_5ForCausalLM)
        # when appropriate.
        if config.get("model_type") in ("qwen35", "qwen3") and _is_qwen35_ssm_gguf(gguf_path):
            config["model_type"] = "qwen3_5_text"
            overrides = _get_qwen35_vision_config_overrides(gguf_path)
            for k, v in overrides.items():
                config.setdefault(k, v)
        return result

    # Patch get_gguf_hf_weights_map to map qwen3_5 → qwen35 for gguf-py arch lookup
    _orig_weights_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_weights_map(hf_model, processor, model_type=None, num_layers=None, **kwargs):
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type in ("qwen3_5", "qwen3_5_text"):
            model_type = "qwen35"
        return _orig_weights_map(hf_model, processor, model_type=model_type, num_layers=num_layers, **kwargs)

    old_gguf_utils = _gguf_utils.load_gguf_checkpoint
    old_config_utils = _config_utils.load_gguf_checkpoint
    old_auto_tokenizer = _auto_tokenizer.load_gguf_checkpoint
    old_tok_utils = _tok_utils.load_gguf_checkpoint
    old_weights_map = _gguf_utils.get_gguf_hf_weights_map

    _gguf_utils.load_gguf_checkpoint = _patched_load
    _config_utils.load_gguf_checkpoint = _patched_load
    _auto_tokenizer.load_gguf_checkpoint = _patched_load
    _tok_utils.load_gguf_checkpoint = _patched_load
    _gguf_utils.get_gguf_hf_weights_map = _patched_weights_map

    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = old_gguf_utils
        _config_utils.load_gguf_checkpoint = old_config_utils
        _auto_tokenizer.load_gguf_checkpoint = old_auto_tokenizer
        _tok_utils.load_gguf_checkpoint = old_tok_utils
        _gguf_utils.get_gguf_hf_weights_map = old_weights_map


class ModelVariant(StrEnum):
    """Available bjivanovich Qwen3.5-4B-Vision GGUF model variants for causal language modeling."""

    QWEN_3_5_4B_VISION_GGUF = "4B_Vision_GGUF"


class ModelLoader(ForgeModel):
    """bjivanovich Qwen3.5-4B-Vision GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_4B_VISION_GGUF: LLMModelConfig(
            pretrained_model_name="bjivanovich/Qwen3.5-4B-Vision-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_4B_VISION_GGUF

    GGUF_FILE = "Qwen3.5-4B.Q4_K_M.gguf"

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
            model="bjivanovich Qwen3.5-4B-Vision GGUF",
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

        with _qwen35_vision_gguf_context():
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
            with _qwen35_vision_gguf_context():
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

        with _qwen35_vision_gguf_context():
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
        with _qwen35_vision_gguf_context():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
