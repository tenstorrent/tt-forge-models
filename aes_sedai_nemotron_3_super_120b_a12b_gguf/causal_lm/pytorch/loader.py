# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling.
"""
import sys
import types
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
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

# The GGUF file uses 'nemotron_h_moe' but transformers knows this architecture
# as 'nemotron_h' (using custom code via auto_map from the base model repo).
_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def _install_mamba_ssm_stub():
    """Inject a pure-PyTorch stub for mamba_ssm into sys.modules.

    modeling_nemotron_h.py imports rmsnorm_fn from mamba_ssm at module load
    time.  The real package requires CUDA/Triton kernels which are not
    available on Tenstorrent hardware.  We provide a pure-PyTorch fallback
    that is sufficient for model construction and XLA compilation.
    """
    if "mamba_ssm" in sys.modules:
        return

    def _rmsnorm_fn(
        x,
        weight,
        bias=None,
        z=None,
        eps=1e-5,
        group_size=None,
        norm_before_gate=False,
    ):
        """Grouped RMS norm with optional SiLU gating (pure PyTorch)."""
        orig_dtype = x.dtype
        x = x.float()

        if group_size is not None and group_size < x.shape[-1]:
            orig_shape = x.shape
            x = x.view(*orig_shape[:-1], orig_shape[-1] // group_size, group_size)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            x = x.view(orig_shape)
        else:
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)

        if weight is not None:
            x = x * weight.float()
        if bias is not None:
            x = x + bias.float()

        if z is not None:
            gate = torch.nn.functional.silu(z.float())
            if norm_before_gate:
                x = x * gate
            else:
                x = x * gate

        return x.to(orig_dtype)

    import importlib.util

    def _make_stub(name):
        mod = types.ModuleType(name)
        # importlib.util.find_spec requires __spec__ to be set; provide a
        # minimal ModuleSpec so that is_mamba_2_ssm_available() does not raise.
        mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        mod.__version__ = "0.0.0"
        return mod

    # Build the nested module hierarchy that modeling_nemotron_h.py expects.
    mamba_ssm_mod = _make_stub("mamba_ssm")
    ops_mod = _make_stub("mamba_ssm.ops")
    triton_mod = _make_stub("mamba_ssm.ops.triton")
    lg_mod = _make_stub("mamba_ssm.ops.triton.layernorm_gated")
    ssd_mod = _make_stub("mamba_ssm.ops.triton.ssd_combined")
    ssu_mod = _make_stub("mamba_ssm.ops.triton.selective_state_update")

    lg_mod.rmsnorm_fn = _rmsnorm_fn
    ssd_mod.mamba_chunk_scan_combined = None
    ssd_mod.mamba_split_conv1d_scan_combined = None
    ssu_mod.selective_state_update = None

    mamba_ssm_mod.ops = ops_mod
    ops_mod.triton = triton_mod
    triton_mod.layernorm_gated = lg_mod
    triton_mod.ssd_combined = ssd_mod
    triton_mod.selective_state_update = ssu_mod

    sys.modules["mamba_ssm"] = mamba_ssm_mod
    sys.modules["mamba_ssm.ops"] = ops_mod
    sys.modules["mamba_ssm.ops.triton"] = triton_mod
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"] = lg_mod
    sys.modules["mamba_ssm.ops.triton.ssd_combined"] = ssd_mod
    sys.modules["mamba_ssm.ops.triton.selective_state_update"] = ssu_mod


def _patch_nemotron_h_moe_support():
    """Register nemotron_h_moe as a supported GGUF architecture.

    The GGUF file declares the architecture as 'nemotron_h_moe', but transformers
    uses the name 'nemotron_h' with custom code. We register the GGUF arch name
    and provide field mappings so load_gguf_checkpoint can parse the KV pairs.
    """
    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    # Field mappings: GGUF key suffix → transformers config field name.
    # These are based on the KV pairs in the Q4_K_M GGUF file.
    config_mapping = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "embedding_length": "hidden_size",
        "vocab_size": "vocab_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.key_length": "head_dim",
        "attention.layer_norm_rms_epsilon": "norm_eps",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        # MoE parameters
        "expert_count": "n_routed_experts",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_shared_count": "n_shared_experts",
        "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
        "expert_used_count": "num_experts_per_tok",
        "expert_weights_norm": "norm_topk_prob",
        "expert_weights_scale": "routed_scaling_factor",
        "moe_latent_size": "moe_latent_size",
        # SSM / Mamba parameters
        "ssm.conv_kernel": "conv_kernel",
        "ssm.group_count": "n_groups",
        "ssm.state_size": "ssm_state_size",
        "ssm.time_step_rank": "mamba_head_dim",
    }
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
            "nemotron_h_moe"
        ] = config_mapping

    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["nemotron_h_moe"] = {}

    # The tokenizer converter uses model_type (nemotron_h) as the key.
    # Map it to the nemotron converter since the tokenizer family is the same.
    if "nemotron" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h"] = GGUF_TO_FAST_CONVERTERS["nemotron"]


def _register_nemotron_h_classes():
    """Download and register custom NemotronH classes with the AutoModel registry.

    The NemotronH architecture is not part of standard transformers; it lives in
    the base model repo. We download it once and register the classes so that
    AutoConfig / AutoModelForCausalLM can find them without needing
    trust_remote_code on every subsequent call.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "nemotron_h" in CONFIG_MAPPING:
        return

    # Install mamba_ssm stub before importing the model class so that the
    # module-level import in modeling_nemotron_h.py succeeds on non-CUDA hosts.
    _install_mamba_ssm_stub()

    # Download config + code from the gated base model.
    base_cfg = AutoConfig.from_pretrained(_BASE_MODEL, trust_remote_code=True)
    cfg_cls = type(base_cfg)
    AutoConfig.register("nemotron_h", cfg_cls, exist_ok=True)

    # Load model class via transformers' dynamic module utilities.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_cls = get_class_from_dynamic_module(
        "modeling_nemotron_h.NemotronHForCausalLM",
        _BASE_MODEL,
    )
    AutoModelForCausalLM.register(cfg_cls, model_cls, exist_ok=True)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    """Wrap load_gguf_checkpoint to add nemotron_h_moe support."""
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
    if result.get("config", {}).get("model_type") == "nemotron_h_moe":
        result["config"]["model_type"] = "nemotron_h"
    return result


_patch_nemotron_h_moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model variants for causal language modeling."""

    AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF = "AesSedai_3_Super_120B_A12B_GGUF"


class ModelLoader(ForgeModel):
    """AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF

    GGUF_FILE = (
        "Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M-00001-of-00003.gguf"
    )

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
            model="AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF",
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
        _register_nemotron_h_classes()

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs["trust_remote_code"] = True

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                gguf_file=self.GGUF_FILE,
                trust_remote_code=True,
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
            trust_remote_code=True,
        )
        return self.config
