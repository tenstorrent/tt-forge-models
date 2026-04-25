# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling.
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

# hybrid_override_pattern for NVIDIA-Nemotron-3-Super-120B (88 layers: M=mamba, E=moe, *=attention)
_HYBRID_OVERRIDE_PATTERN_120B = (
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
    "EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
)

# HuggingFace repo containing the NemotronH model code (trust_remote_code)
_NEMOTRON_H_CODE_REPO = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def _patch_transformers_nemotron_h_moe_gguf():
    """Monkey-patch transformers to add nemotron_h_moe GGUF architecture support.

    The GGUF file uses architecture name nemotron_h_moe, but transformers only
    knows about nemotron_h (via trust_remote_code). This patch registers the
    architecture and rewrites the model_type after config parsing.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        NemotronTensorProcessor,
        TENSOR_PROCESSORS,
    )

    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["nemotron_h_moe"] = {
        "block_count": "num_hidden_layers",
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "feed_forward_length": "intermediate_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "rope.freq_base": "rope_theta",
        "rope.dimension_count": None,
        "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
        "attention.layer_norm_epsilon": None,
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_shared_count": "n_shared_experts",
        "expert_weights_scale": "routed_scaling_factor",
        "expert_weights_norm": "norm_topk_prob",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.state_size": "ssm_state_size",
        "ssm.group_count": "mamba_n_groups",
        "ssm.inner_size": None,
        "ssm.time_step_rank": None,
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "moe_latent_size": "moe_latent_size",
    }

    TENSOR_PROCESSORS["nemotron_h_moe"] = NemotronTensorProcessor

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

    if "nemotron_h_moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUFGPTConverter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "nemotron_h_moe":
            config["model_type"] = "nemotron_h"
            config["hybrid_override_pattern"] = _HYBRID_OVERRIDE_PATTERN_120B
            # GGUF reports 0 KV heads; actual model has 2
            if config.get("num_key_value_heads", 0) == 0:
                config["num_key_value_heads"] = 2
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


def _ensure_mamba_ssm_not_broken():
    """Patch mamba_ssm stubs so NemotronH can be imported without real CUDA mamba-ssm.

    Other loaders (e.g. plantcad2) install mamba_ssm stub modules via types.ModuleType
    with __spec__=None.  importlib.util.find_spec raises ValueError for such entries,
    breaking is_mamba_2_ssm_available() inside modeling_nemotron_h.py at import time.

    Additionally, modeling_nemotron_h.py unconditionally does:
        from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    which is not in the plantcad2 stubs, causing an ImportError.

    Fix:
      1. Replace is_mamba_2_ssm_available / is_mamba_ssm_available with lambdas that
         return False without calling find_spec.
      2. Inject a minimal mamba_ssm.ops.triton.layernorm_gated stub with rmsnorm_fn.
    """
    import sys
    import types
    import torch.nn.functional as F
    import transformers.utils.import_utils as _iu

    # Replace the lru_cache-wrapped functions with plain lambdas so that
    # modeling_nemotron_h.py's "from transformers.utils.import_utils import ..."
    # picks up False-returning versions (avoids ValueError from find_spec).
    _iu.is_mamba_2_ssm_available = lambda: False
    _iu.is_mamba_ssm_available = lambda: False

    # Ensure parent mamba_ssm stub packages exist (plantcad2 may have installed them)
    def _ensure_stub(name, is_pkg=True):
        if name not in sys.modules:
            m = types.ModuleType(name)
            if is_pkg:
                m.__path__ = []
            sys.modules[name] = m
        return sys.modules[name]

    _ensure_stub("mamba_ssm")
    _ensure_stub("mamba_ssm.ops")
    _ensure_stub("mamba_ssm.ops.triton")

    # Add layernorm_gated stub required unconditionally by modeling_nemotron_h.py
    if "mamba_ssm.ops.triton.layernorm_gated" not in sys.modules:
        ln_mod = types.ModuleType("mamba_ssm.ops.triton.layernorm_gated")

        def rmsnorm_fn(
            x,
            weight,
            bias=None,
            residual=None,
            eps=1e-6,
            prenorm=False,
            residual_in_fp32=False,
        ):
            if residual is not None:
                x = x + residual
            out = F.rms_norm(x, weight.shape, weight, eps)
            if prenorm:
                return out, x
            return out

        ln_mod.rmsnorm_fn = rmsnorm_fn
        sys.modules["mamba_ssm.ops.triton.layernorm_gated"] = ln_mod


_patch_transformers_nemotron_h_moe_gguf()


class ModelVariant(StrEnum):
    """Available Nemotron 3 Super 120B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF = (
        "3_Super_120B_A12B_BF16_heretic_i1_GGUF"
    )
    GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF = "ggml_org_3_Super_120B_GGUF"


class ModelLoader(ForgeModel):
    """Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: LLMModelConfig(
            pretrained_model_name="ggml-org/Nemotron-3-Super-120B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: "NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic.i1-Q4_K_M.gguf",
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: "Nemotron-3-Super-120B-Q4_K.gguf",
    }

    sample_text = "Give me a short introduction to large language model."

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

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
            model="Nemotron 3 Super 120B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        # Load tokenizer from the nvidia model: nemotron_h requires trust_remote_code
        # and AutoTokenizer cannot resolve it from the GGUF file alone.
        self.tokenizer = AutoTokenizer.from_pretrained(
            _NEMOTRON_H_CODE_REPO,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _load_nemotron_h_config(self):
        """Load NemotronH config from the original nvidia model (trust_remote_code)."""
        # Clean up any broken mamba_ssm stub before loading the dynamic module
        _ensure_mamba_ssm_not_broken()
        config = AutoConfig.from_pretrained(
            _NEMOTRON_H_CODE_REPO,
            trust_remote_code=True,
        )
        # Disable CUDA-only Mamba kernels for CPU/XLA compilation
        config.use_mamba_kernels = False
        if self.num_layers is not None:
            # Trim hybrid_override_pattern to first num_layers characters
            pattern = config.hybrid_override_pattern[: self.num_layers]
            config.hybrid_override_pattern = pattern
        return config

    def _get_gguf_local_path(self):
        """Return the local cached path of the GGUF file."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            self._variant_config.pretrained_model_name,
            filename=self.gguf_file,
            local_files_only=True,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        # Pass the local GGUF path so transformers resolves weights without
        # looking for model code in the GGUF-only mradermacher repo.
        model_kwargs["gguf_file"] = self._get_gguf_local_path()
        model_kwargs["trust_remote_code"] = True
        model_kwargs["ignore_mismatched_sizes"] = True

        # Load config from the nvidia code repo so NemotronHForCausalLM is available
        model_kwargs["config"] = self._load_nemotron_h_config()

        # Clean up broken mamba_ssm just before the dynamic module is imported
        _ensure_mamba_ssm_not_broken()
        # Use the nvidia model repo as the pretrained source so transformers can
        # find modeling_nemotron_h.py; weights come from the local GGUF file.
        model = AutoModelForCausalLM.from_pretrained(
            _NEMOTRON_H_CODE_REPO, **model_kwargs
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

    def load_config(self):
        self.config = self._load_nemotron_h_config()
        return self.config
