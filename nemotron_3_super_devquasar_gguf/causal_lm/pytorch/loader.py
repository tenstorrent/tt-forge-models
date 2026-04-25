# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DevQuasar NVIDIA Nemotron 3 Super 120B A12B BF16 GGUF model loader implementation for causal language modeling.
"""
import inspect
import re
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import PreTrainedModel
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUFTensor,
    TensorProcessor,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
    TENSOR_PROCESSORS,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

# --- NemotronH-MoE GGUF support patch ---
#
# 1. Register NemotronHConfig / NemotronHForCausalLM from the NVIDIA repo so
#    that AutoConfig.for_model("nemotron_h") works without trust_remote_code at
#    the call site (the classes are pre-loaded once here).
# 2. Register "nemotron_h_moe" as a recognised GGUF architecture so that
#    load_gguf_checkpoint accepts the GGUF file produced for this model.
# 3. A custom TensorProcessor renames HF tensor paths to match the
#    gguf-py TensorNameMap for NEMOTRON_H_MOE, and handles splitting the
#    stacked per-expert weight tensors.


_NEMOTRON_H_MOE_GGUF_ARCH = "nemotron_h_moe"
_NVIDIA_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

# Mapping: GGUF nemotron_h_moe field names -> transformers config keys.
_NEMOTRON_H_MOE_CONFIG_MAP = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
    "vocab_size": "vocab_size",
    "expert_count": "n_routed_experts",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_used_count": "num_experts_per_tok",
    "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
    "expert_weights_scale": "routed_scaling_factor",
    "moe_latent_size": "moe_latent_size",
    "ssm.state_size": "ssm_state_size",
    "ssm.conv_kernel": "conv_kernel",
    "ssm.inner_size": None,
}


def _register_nemotron_h_classes():
    """Load and register NemotronHConfig/NemotronHForCausalLM once."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "nemotron_h" in CONFIG_MAPPING:
        return  # already registered

    try:
        # Pull the config class from the NVIDIA model repo.
        _nvidia_cfg_cls = AutoConfig.from_pretrained(
            _NVIDIA_MODEL_ID, trust_remote_code=True
        ).__class__

        try:
            AutoConfig.register("nemotron_h", _nvidia_cfg_cls, exist_ok=True)
        except Exception:
            pass

        # Pull the model class by temporarily loading model code.
        _nvidia_model_cls = AutoModelForCausalLM._model_mapping.get(_nvidia_cfg_cls)
        if _nvidia_model_cls is None:
            # Force model code download.
            import importlib
            from huggingface_hub import hf_hub_download

            modeling_path = hf_hub_download(_NVIDIA_MODEL_ID, "modeling_nemotron_h.py")
            import importlib.util as _ilu

            spec = _ilu.spec_from_file_location("modeling_nemotron_h", modeling_path)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _nvidia_model_cls = mod.NemotronHForCausalLM

        if _nvidia_model_cls is not None:
            try:
                AutoModelForCausalLM.register(
                    _nvidia_cfg_cls, _nvidia_model_cls, exist_ok=True
                )
            except Exception:
                pass
    except Exception:
        pass


def _patch_nemotron_h_moe_gguf():
    """Register nemotron_h_moe in transformers' GGUF architecture tables."""
    if _NEMOTRON_H_MOE_GGUF_ARCH in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append(_NEMOTRON_H_MOE_GGUF_ARCH)

    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "nemotron" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            base = dict(_gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["nemotron"])
            base.update(_NEMOTRON_H_MOE_CONFIG_MAP)
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
                _NEMOTRON_H_MOE_GGUF_ARCH
            ] = base

    if "nemotron" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS[_NEMOTRON_H_MOE_GGUF_ARCH] = GGUF_TO_FAST_CONVERTERS[
            "nemotron"
        ]
        # The tokenizer loader uses config["model_type"] as architecture key,
        # which our patch renames from "nemotron_h_moe" to "nemotron_h".
        GGUF_TO_FAST_CONVERTERS["nemotron_h"] = GGUF_TO_FAST_CONVERTERS["nemotron"]

    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "nemotron" in _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING[
                _NEMOTRON_H_MOE_GGUF_ARCH
            ] = _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["nemotron"]


class NemotronHMoeTensorProcessor(TensorProcessor):
    """Tensor processor for NemotronH-MoE GGUF checkpoints.

    Maps HF state-dict names (which use 'model.layers.N.mixer.*') to the
    GGUF gguf-py TensorNameMap keys (which use 'backbone.layers.N.mixer.*'),
    and handles splitting stacked GGUF expert-weight tensors into per-expert
    entries.
    """

    _STACKED_EXPERT_PAT = re.compile(
        r"^blk\.(?P<bid>\d+)\.ffn_(?P<w>up|down)_exps(\.weight)?$"
    )
    _HF_EXPERT_PAT = re.compile(
        r"^backbone\.layers\.(?P<bid>\d+)\.mixer\.experts\.(?P<eid>\d+)\."
        r"(?P<w>up|down)_proj$"
    )

    def preprocess_name(self, hf_name: str) -> str:
        hf_name = re.sub(r"^model\.layers\.", "backbone.layers.", hf_name)
        hf_name = re.sub(r"^model\.embeddings\.", "backbone.embedding.", hf_name)
        hf_name = re.sub(r"^model\.norm_f\.", "backbone.norm_f.", hf_name)
        return hf_name

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map: dict, suffix: str, qual_name: str, hf_name: str
    ):
        pre = self.preprocess_name(hf_name)
        name_no_sfx = pre
        if pre.endswith(".weight") or pre.endswith(".bias"):
            name_no_sfx = pre.rsplit(".", 1)[0]
        if m := self._HF_EXPERT_PAT.match(name_no_sfx):
            gguf_key = f"blk.{m['bid']}.ffn_{m['w']}_exps{suffix}"
            full_hf = qual_name + hf_name
            if gguf_key not in gguf_to_hf_name_map:
                gguf_to_hf_name_map[gguf_key] = []
            if isinstance(gguf_to_hf_name_map[gguf_key], list):
                gguf_to_hf_name_map[gguf_key].append((int(m["eid"]), full_hf))

    def process(self, weights, name: str, **kwargs):
        if m := self._STACKED_EXPERT_PAT.match(name):
            tensor_key_mapping = kwargs.get("tensor_key_mapping", {})
            parsed_parameters = kwargs.get("parsed_parameters", {})
            tensors = parsed_parameters.get("tensors", {})
            # Try both with and without .weight suffix.
            entries = tensor_key_mapping.get(
                name + ".weight"
            ) or tensor_key_mapping.get(name)
            if isinstance(entries, list):
                arr = torch.from_numpy(np.copy(weights))
                for expert_idx, hf_key in sorted(entries):
                    if expert_idx < arr.shape[0]:
                        tensors[hf_key] = arr[expert_idx].clone()
                return GGUFTensor(weights, None, {})
        return GGUFTensor(weights, name, {})


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to support nemotron_h_moe architecture."""
    _patch_nemotron_h_moe_gguf()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    cfg = result.get("config", {})
    if cfg.get("model_type") == _NEMOTRON_H_MOE_GGUF_ARCH:
        cfg["model_type"] = "nemotron_h"
        cfg["architectures"] = ["NemotronHForCausalLM"]
        # num_key_value_heads is stored as a per-layer list; take first non-zero.
        kv = cfg.get("num_key_value_heads")
        if isinstance(kv, list):
            non_zero = [h for h in kv if h > 0]
            cfg["num_key_value_heads"] = non_zero[0] if non_zero else 2
        # Supplement missing SSM/hybrid fields from the NVIDIA reference config.
        try:
            ref = AutoConfig.from_pretrained(_NVIDIA_MODEL_ID, trust_remote_code=True)
            _SUPPLEMENT = [
                "layers_block_type",
                "hybrid_override_pattern",
                "mamba_num_heads",
                "mamba_head_dim",
                "mamba_hidden_act",
                "mamba_proj_bias",
                "mamba_ssm_cache_dtype",
                "expand",
                "chunk_size",
                "head_dim",
                "n_groups",
                "n_group",
                "topk_group",
                "norm_topk_prob",
                "n_shared_experts",
                "moe_shared_expert_overlap",
                "mlp_bias",
                "mlp_hidden_act",
                "attention_bias",
                "partial_rotary_factor",
                "rescale_prenorm_residual",
                "residual_in_fp32",
                "use_bias",
                "use_conv_bias",
                "use_mamba_kernels",
                "use_cache",
                "time_step_min",
                "time_step_max",
                "time_step_floor",
                "norm_eps",
                "intermediate_size",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "tie_word_embeddings",
                "num_nextn_predict_layers",
                "num_logits_to_keep",
            ]
            for f in _SUPPLEMENT:
                if f not in cfg and hasattr(ref, f):
                    cfg[f] = getattr(ref, f)
        except Exception:
            pass
    return result


_patch_nemotron_h_moe_gguf()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

TENSOR_PROCESSORS["nemotron_h_moe"] = NemotronHMoeTensorProcessor
TENSOR_PROCESSORS["nemotron_h"] = NemotronHMoeTensorProcessor

# Patch get_gguf_hf_weights_map to use NEMOTRON_H_MOE arch (which includes
# MoE tensor definitions) when model_type is "nemotron_h".
_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map


def _patched_get_gguf_hf_weights_map(
    hf_model, processor=None, model_type=None, num_layers=None, **kwargs
):
    mt = hf_model.config.model_type if model_type is None else model_type
    if mt == "nemotron_h":
        mt = "nemotron_h_moe"
    orig_params = inspect.signature(_orig_get_gguf_hf_weights_map).parameters
    if "processor" in orig_params:
        return _orig_get_gguf_hf_weights_map(
            hf_model, processor, model_type=mt, num_layers=num_layers, **kwargs
        )
    else:
        return _orig_get_gguf_hf_weights_map(
            hf_model, model_type=mt, num_layers=num_layers, **kwargs
        )


_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

# Register NemotronH classes from the NVIDIA repo into the Auto* registries.
_register_nemotron_h_classes()

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
    """Available DevQuasar NVIDIA Nemotron 3 Super 120B A12B BF16 GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_BF16_Q4_K_M_GGUF = "120B_A12B_BF16_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """DevQuasar NVIDIA Nemotron 3 Super 120B A12B BF16 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="DevQuasar/nvidia.NVIDIA-Nemotron-3-Super-120B-A12B-BF16-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_Q4_K_M_GGUF

    GGUF_FILE = (
        "nvidia.NVIDIA-Nemotron-3-Super-120B-A12B-BF16.Q4_K_M-00001-of-00006.gguf"
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
            model="Nemotron 3 Super DevQuasar GGUF",
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

    def _get_gguf_local_path(self):
        """Return the local cached path to the GGUF first shard."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            self._variant_config.pretrained_model_name,
            self.GGUF_FILE,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load config from NVIDIA repo (has NemotronHConfig custom code).
        config = AutoConfig.from_pretrained(_NVIDIA_MODEL_ID, trust_remote_code=True)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
            config.layers_block_type = config.layers_block_type[: self.num_layers]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["trust_remote_code"] = True
        model_kwargs["config"] = config

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            # Load model from NVIDIA repo with DevQuasar GGUF weights.
            gguf_local = self._get_gguf_local_path()
            model_kwargs["gguf_file"] = gguf_local
            model = AutoModelForCausalLM.from_pretrained(
                _NVIDIA_MODEL_ID, **model_kwargs
            )

        model.eval()
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
            if hasattr(layer.mixer, "q_proj"):
                shard_specs[layer.mixer.q_proj.weight] = ("model", "batch")
                shard_specs[layer.mixer.k_proj.weight] = ("model", "batch")
                shard_specs[layer.mixer.v_proj.weight] = ("model", "batch")
                shard_specs[layer.mixer.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        if self.config is not None:
            return self.config
        self.config = AutoConfig.from_pretrained(
            _NVIDIA_MODEL_ID, trust_remote_code=True
        )
        return self.config
