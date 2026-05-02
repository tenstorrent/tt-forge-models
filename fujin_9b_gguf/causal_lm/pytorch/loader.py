# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fujin 9B i1 GGUF model loader implementation for causal language modeling.
"""
import gc
import re
import contextlib
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

# Config field mapping for the qwen35 (Qwen3.5 SSM hybrid) GGUF architecture.
# The gguf library already has a complete tensor-name map for MODEL_ARCH.QWEN35,
# so only the metadata → HF config field translation is needed here.
_QWEN35_CONFIG_FIELDS = {
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
    # Passed as **kwargs to Qwen3_5TextConfig which uses it to auto-generate layer_types.
    "full_attention_interval": "full_attention_interval",
}

# Translation from HF model_type ("qwen3_5_text") back to gguf-py arch name ("qwen35").
# get_gguf_hf_weights_map looks up model_type in gguf-py MODEL_ARCH_NAMES which only
# knows the GGUF-native names.
_QWEN35_HF_TO_GGUF_TYPE = {"qwen3_5_text": "qwen35", "qwen3_5": "qwen35"}


class _Qwen35TensorProcessor:
    """
    Minimal TensorProcessor for the Qwen3.5 SSM hybrid (qwen35) GGUF architecture.

    Fixes two gaps left by the gguf-py tensor-name map:
    - ssm_dt.bias is not mapped by gguf-py; add it via perform_fallback_tensor_mapping.
    - ssm_conv1d.weight is dequantized to (H, K) but the HF model expects (H, 1, K);
      unsqueeze axis 1 in process().
    """

    def __init__(self, config=None):
        self.config = config or {}

    def preprocess_name(self, hf_name: str) -> str:
        return hf_name

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map: dict, suffix: str, qual_name: str, hf_name: str
    ):
        # blk.N.ssm_dt.bias → model.layers.N.linear_attn.dt_bias
        m = re.match(r"model\.layers\.(\d+)\.linear_attn\.dt_bias", hf_name)
        if m:
            layer_idx = m.group(1)
            gguf_key = f"blk.{layer_idx}.ssm_dt.bias"
            gguf_to_hf_name_map[gguf_key] = qual_name + hf_name

    def process(self, weights, name, **kwargs):
        from transformers.modeling_gguf_pytorch_utils import GGUFTensor

        # conv1d weights are dequantized to (H, K) but model.conv1d expects (H, 1, K).
        if "ssm_conv1d" in name:
            weights = np.expand_dims(weights, axis=1)
        return GGUFTensor(weights, name, {})


def _find_real(module_name: str, func_name: str):
    """Return the unpatched version of func_name from module_name via gc scan."""
    for obj in gc.get_objects():
        try:
            if (
                callable(obj)
                and getattr(obj, "__name__", None) == func_name
                and getattr(obj, "__module__", None) == module_name
            ):
                return obj
        except Exception:
            pass
    return None


@contextlib.contextmanager
def _qwen35_gguf_ctx():
    """
    Prepare the transformers GGUF stack to load a qwen35 (Qwen3.5 SSM hybrid) checkpoint.

    Addresses four problems:
    1. Collection-time loaders may replace load_gguf_checkpoint with a patch that
       lacks the model_to_load kwarg (transformers 5.x).  Restore the real function
       via gc scan.
    2. qwen35 is absent from GGUF_CONFIG_MAPPING; add the field mapping temporarily.
    3. load_gguf_checkpoint returns model_type="qwen35"; transformers expects
       "qwen3_5_text".  Fix in the wrapper.
    4. get_gguf_hf_weights_map looks up model_type in gguf-py MODEL_ARCH_NAMES which
       has "qwen35" but not "qwen3_5_text".  Translate before the lookup.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.configuration_utils as _config_utils
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
    from transformers.modeling_gguf_pytorch_utils import TENSOR_PROCESSORS

    _MOD = "transformers.modeling_gguf_pytorch_utils"
    real_load = _find_real(_MOD, "load_gguf_checkpoint")
    real_get_map = _find_real(_MOD, "get_gguf_hf_weights_map")

    # Temporarily add qwen35 config-field mapping and tensor processor.
    had_cfg = "qwen35" in GGUF_CONFIG_MAPPING
    had_proc = "qwen35" in TENSOR_PROCESSORS
    if not had_cfg:
        GGUF_CONFIG_MAPPING["qwen35"] = _QWEN35_CONFIG_FIELDS
    if not had_proc:
        TENSOR_PROCESSORS["qwen35"] = _Qwen35TensorProcessor

    def _wrapped_load(gguf_path, return_tensors=False, model_to_load=None, **kwargs):
        fn = real_load if real_load is not None else _gguf_utils.load_gguf_checkpoint
        result = fn(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen35":
            result["config"]["model_type"] = "qwen3_5_text"
        return result

    def _wrapped_get_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        if model_type is None and hasattr(hf_model, "config"):
            mt = getattr(hf_model.config, "model_type", None)
            model_type = _QWEN35_HF_TO_GGUF_TYPE.get(mt, mt)
        elif model_type in _QWEN35_HF_TO_GGUF_TYPE:
            model_type = _QWEN35_HF_TO_GGUF_TYPE[model_type]
        fn = real_get_map if real_get_map is not None else _gguf_utils.get_gguf_hf_weights_map
        return fn(hf_model, processor, model_type, num_layers, qual_name)

    saved_load_gguf = _gguf_utils.load_gguf_checkpoint
    saved_cfg_load = _config_utils.load_gguf_checkpoint
    saved_get_map = _gguf_utils.get_gguf_hf_weights_map
    _gguf_utils.load_gguf_checkpoint = _wrapped_load
    _config_utils.load_gguf_checkpoint = _wrapped_load
    _gguf_utils.get_gguf_hf_weights_map = _wrapped_get_map
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = saved_load_gguf
        _config_utils.load_gguf_checkpoint = saved_cfg_load
        _gguf_utils.get_gguf_hf_weights_map = saved_get_map
        if not had_cfg:
            GGUF_CONFIG_MAPPING.pop("qwen35", None)
        if not had_proc:
            TENSOR_PROCESSORS.pop("qwen35", None)


class ModelVariant(StrEnum):
    """Available Fujin 9B i1 GGUF model variants for causal language modeling."""

    FUJIN_9B_I1_Q4_K_M_GGUF = "FUJIN_9B_I1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Fujin 9B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FUJIN_9B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/fujin-9b-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FUJIN_9B_I1_Q4_K_M_GGUF

    GGUF_FILE = "fujin-9b.i1-Q4_K_M.gguf"

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
            model="Fujin 9B i1 GGUF",
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
            with _qwen35_gguf_ctx():
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

        with _qwen35_gguf_ctx():
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
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text
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
        """Get the text config, handling both nested and flat config structures."""
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
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        with _qwen35_gguf_ctx():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
                gguf_file=self.GGUF_FILE,
            )
        return self.config
