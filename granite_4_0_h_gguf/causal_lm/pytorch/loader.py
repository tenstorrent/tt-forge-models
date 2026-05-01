# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite 4.0 H-Tiny GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
import re
import types

import numpy as np
import torch
import torch.nn.functional as F
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


def _patched_topk_gating_forward(self, hidden_states):
    """Avoids expert_size.tolist() by returning sorted_expert_ids as a tensor.

    The original GraniteMoeHybridTopKGating.forward calls expert_size.tolist()
    which triggers a device-to-host transfer that fails on TT silicon (INTERNAL
    error code 13). Instead we return sorted_expert_ids as an int32 tensor so
    the caller (GraniteMoeHybridParallelExperts) can use it with weight indexing.
    """
    logits = self.layer(hidden_states).float()
    top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
    top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)

    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")

    # Return sorted_expert_ids as int32 tensor instead of expert_size list.
    sorted_expert_ids = top_k_experts[index_sorted_experts].int()

    top_k_gates = top_k_gates.flatten()
    batch_gates = top_k_gates[index_sorted_experts]

    return index_sorted_experts, batch_index, batch_gates, sorted_expert_ids, logits


def _patched_parallel_experts_forward(self, inputs, sorted_expert_ids):
    """Uses per-expert masked matmul instead of split-by-expert-size or gather.

    The original GraniteMoeHybridParallelExperts.forward calls
    inputs.split(expert_size) with a Python list, requiring a device-to-host
    transfer that fails on TT silicon.

    A simple weight gather (self.weight[sorted_expert_ids]) is also
    problematic: MLIR flattens the 3D weight [num_experts, output_size,
    input_size] to a 2D embedding table [num_experts, output_size*input_size],
    whose row size (~3 MB) overflows the L1 CB budget on TT silicon.

    Instead: for each expert e (statically unrolled at trace time), compute
    F.linear for all tokens and zero-out tokens not assigned to e via a
    boolean mask. The per-expert weight slices self.weight[e] are constant
    integer-indexed and never create a dynamic gather/embedding. All ops
    stay in tensor-land with no Python-level splits or device-to-host transfers.
    """
    T = inputs.shape[0]
    result = torch.zeros(T, self.output_size, dtype=inputs.dtype, device=inputs.device)
    for e in range(self.num_experts):
        w_e = self.weight[e]  # [output_size, input_size] — static slice
        out_e = F.linear(inputs, w_e)  # [T, output_size]
        mask_e = (sorted_expert_ids == e).to(inputs.dtype).unsqueeze(1)
        result = result + out_e * mask_e
    return result


def _patch_moe_experts(model):
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridParallelExperts,
        GraniteMoeHybridTopKGating,
    )

    for module in model.modules():
        if isinstance(module, GraniteMoeHybridTopKGating):
            module.forward = types.MethodType(_patched_topk_gating_forward, module)
        elif isinstance(module, GraniteMoeHybridParallelExperts):
            module.forward = types.MethodType(
                _patched_parallel_experts_forward, module
            )


def _patch_transformers_granitehybrid_gguf():
    """Monkey-patch transformers to add granitehybrid GGUF architecture support.

    Transformers 5.x has GraniteMoeHybridForCausalLM but lacks GGUF loading
    support for the granitehybrid architecture. This patch:
      1. Registers granitehybrid in GGUF_CONFIG_MAPPING / GGUF_TO_FAST_CONVERTERS.
      2. Patches load_gguf_checkpoint to remap model_type and derive layer_types.
      3. Registers a TensorProcessor that:
         - maps blk.N.ffn_gate_exps + ffn_up_exps → block_sparse_moe.input_linear
           (concatenated along expert-intermediate dim, transposed to [E, 2I, H])
         - maps blk.N.ffn_gate_shexp + ffn_up_shexp → shared_mlp.input_linear
           (concatenated along intermediate dim, transposed to [2I, H])
         - fixes Mamba2 tensor shapes (ssm_conv1d, ssm_a/ssm_d)
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TensorProcessor,
        GGUFTensor,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "granitehybrid" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    # 1. Register architecture in GGUF config mapping
    GGUF_SUPPORTED_ARCHITECTURES.append("granitehybrid")
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["granitehybrid"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    # 2. Register tokenizer converter (GPT-2 style BPE)
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter
    for name in ("granitehybrid", "granitemoehybrid"):
        if name not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS[name] = GGUFGPTConverter

    # 3. Register TensorProcessor for granitehybrid.
    #
    # GGUF format for granitehybrid expert FFN tensors:
    #   blk.N.ffn_gate_exps.weight : [hidden, intermediate, num_experts]
    #   blk.N.ffn_up_exps.weight   : [hidden, intermediate, num_experts]
    #   blk.N.ffn_down_exps.weight : [intermediate, hidden, num_experts]  → gguf-py handles this
    #   blk.N.ffn_gate_inp.weight  : [hidden, num_experts]                → gguf-py handles this
    # GGUF format for shared MLP tensors:
    #   blk.N.ffn_gate_shexp.weight : [hidden, intermediate]
    #   blk.N.ffn_up_shexp.weight   : [hidden, intermediate]
    #   blk.N.ffn_down_shexp.weight : [intermediate, hidden]              → gguf-py handles this
    # GGUF format for Mamba2 tensors:
    #   blk.N.ssm_conv1d.weight : [kernel, channels]   → HF [channels, 1, kernel]
    #   blk.N.ssm_a             : [1, n_heads]          → HF [n_heads]
    #   blk.N.ssm_d             : [1, n_heads]          → HF [n_heads]
    #
    # HF format:
    #   block_sparse_moe.input_linear.weight  : [num_experts, intermediate*2, hidden]
    #   shared_mlp.input_linear.weight        : [intermediate*2, hidden]
    class _GraniteHybridTensorProcessor(TensorProcessor):
        # Pattern for expert gate/up tensors (many-to-one mapping)
        _EXPS_GATE_UP_RE = re.compile(
            r"blk\.(?P<bid>\d+)\.ffn_(?P<w>gate|up)_exps\.weight$"
        )
        # Pattern for shared-MLP gate/up tensors (many-to-one mapping)
        _SHEXP_GATE_UP_RE = re.compile(
            r"blk\.(?P<bid>\d+)\.ffn_(?P<w>gate|up)_shexp\.weight$"
        )
        # HF name patterns that need fallback mapping.
        # Note: perform_fallback_tensor_mapping receives the FULL hf_name including suffix
        # (e.g., 'model.layers.0.block_sparse_moe.input_linear.weight')
        _HF_MOE_INPUT_RE = re.compile(
            r"model\.layers\.(?P<bid>\d+)\.block_sparse_moe\.input_linear(?P<sfx>\.\w+)$"
        )
        _HF_SHMLP_INPUT_RE = re.compile(
            r"model\.layers\.(?P<bid>\d+)\.shared_mlp\.input_linear(?P<sfx>\.\w+)$"
        )
        _HF_DT_BIAS_RE = re.compile(
            r"model\.layers\.(?P<bid>\d+)\.mamba\.dt_bias$"
        )

        def perform_fallback_tensor_mapping(
            self,
            gguf_to_hf_name_map: dict,
            suffix: str,
            qual_name: str,
            hf_name: str,
        ):
            # block_sparse_moe.input_linear.weight → both ffn_gate_exps and ffn_up_exps
            if m := re.fullmatch(self._HF_MOE_INPUT_RE, hf_name):
                full_hf = qual_name + hf_name
                bid = m["bid"]
                sfx = m["sfx"]
                gguf_to_hf_name_map[f"blk.{bid}.ffn_gate_exps{sfx}"] = full_hf
                gguf_to_hf_name_map[f"blk.{bid}.ffn_up_exps{sfx}"] = full_hf
            # shared_mlp.input_linear.weight → both ffn_gate_shexp and ffn_up_shexp
            elif m := re.fullmatch(self._HF_SHMLP_INPUT_RE, hf_name):
                full_hf = qual_name + hf_name
                bid = m["bid"]
                sfx = m["sfx"]
                gguf_to_hf_name_map[f"blk.{bid}.ffn_gate_shexp{sfx}"] = full_hf
                gguf_to_hf_name_map[f"blk.{bid}.ffn_up_shexp{sfx}"] = full_hf
            # mamba.dt_bias → ssm_dt.bias (gguf-py has no mapping for this).
            # Note: hf_name = "model.layers.N.mamba.dt_bias" (does not end in .bias/.weight)
            # so this is called with suffix="" and hf_name including the full param name.
            elif m := re.fullmatch(self._HF_DT_BIAS_RE, hf_name):
                full_hf = qual_name + hf_name
                bid = m["bid"]
                gguf_to_hf_name_map[f"blk.{bid}.ssm_dt.bias"] = full_hf

        def process(self, weights, name, **kwargs):
            # Mamba2 conv1d: dequantized shape is [channels, kernel]; HF needs [channels, 1, kernel]
            if "ssm_conv1d.weight" in name:
                weights = weights[:, np.newaxis, :]

            # Mamba2 A_log / D: dequantized shape is [n, 1]; HF needs [n]
            if ("ssm_a" in name or "ssm_d" in name) and weights.ndim == 2 and weights.shape[1] == 1:
                weights = weights[:, 0]

            # Expert gate/up: GGUF [hidden, intermediate, num_experts]
            # → accumulate as [num_experts, intermediate, hidden]; concatenate gate+up
            m = re.fullmatch(self._EXPS_GATE_UP_RE, name)
            if m:
                return self._accumulate_gate_up(weights, name, m["w"], "exps", **kwargs)

            # Shared-MLP gate/up: GGUF [hidden, intermediate]
            # → accumulate as [intermediate, hidden]; concatenate gate+up
            m = re.fullmatch(self._SHEXP_GATE_UP_RE, name)
            if m:
                return self._accumulate_gate_up(weights, name, m["w"], "shexp", **kwargs)

            return GGUFTensor(weights, name, {})

        def _accumulate_gate_up(self, weights, name, which, kind, **kwargs):
            """Accumulate gate and up weights into a concatenated input_linear tensor.

            For experts (kind='exps'):
              Dequantized GGUF shape: [num_experts, intermediate, hidden]
              HF target:             [num_experts, intermediate*2, hidden]  (gate first, then up)

            For shared MLP (kind='shexp'):
              Dequantized GGUF shape: [intermediate, hidden]
              HF target:             [intermediate*2, hidden]  (gate first, then up)
            """
            tm = kwargs.get("tensor_key_mapping") or {}
            pp = kwargs.get("parsed_parameters") or {}
            tensors = pp.get("tensors", {})
            hf_name = tm.get(name)
            if not hf_name:
                return GGUFTensor(weights, name, {})

            tw = torch.from_numpy(np.copy(weights))

            if kind == "exps":
                # tw: [E, I, H]; HF target: [E, I*2, H]
                e, i_size, h = tw.shape
                if hf_name not in tensors:
                    tensors[hf_name] = torch.zeros([e, i_size * 2, h], dtype=tw.dtype)
                out = tensors[hf_name]
                if which == "gate":
                    out[:, :i_size, :].copy_(tw)
                else:  # up
                    out[:, i_size:, :].copy_(tw)
            else:  # shexp
                # tw: [I, H]; HF target: [I*2, H]
                i_size, h = tw.shape
                if hf_name not in tensors:
                    tensors[hf_name] = torch.zeros([i_size * 2, h], dtype=tw.dtype)
                out = tensors[hf_name]
                if which == "gate":
                    out[:i_size, :].copy_(tw)
                else:  # up
                    out[i_size:, :].copy_(tw)

            # Return None name to prevent default storage (we've done it manually)
            return GGUFTensor(weights, None, {})

    TENSOR_PROCESSORS["granitehybrid"] = _GraniteHybridTensorProcessor

    # 4. Patch load_gguf_checkpoint: remap model_type + derive layer_types/KV heads
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") != "granitehybrid":
            return result

        config["model_type"] = "granitemoehybrid"
        config["architectures"] = ["GraniteMoeHybridForCausalLM"]

        # Fix num_key_value_heads: GGUF stores per-layer array; take max (attention layers)
        kv = config.get("num_key_value_heads", 0)
        if isinstance(kv, list):
            config["num_key_value_heads"] = max(kv) if kv else 0

        # Parse extra fields (scales, SSM params, MoE) from raw GGUF
        gguf_path = args[0] if args else kwargs.get("gguf_file")
        if gguf_path and (isinstance(gguf_path, (str, bytes)) or hasattr(gguf_path, "__fspath__")):
            try:
                from gguf import GGUFReader
                from transformers.modeling_gguf_pytorch_utils import _gguf_parse_value

                reader = GGUFReader(gguf_path)
                arch = "granitehybrid"
                extra_fields = {
                    f"{arch}.embedding_scale": "embedding_multiplier",
                    f"{arch}.attention.scale": "attention_multiplier",
                    f"{arch}.logit_scale": "logits_scaling",
                    f"{arch}.residual_scale": "residual_multiplier",
                    f"{arch}.ssm.conv_kernel": "mamba_d_conv",
                    f"{arch}.ssm.state_size": "mamba_d_state",
                    f"{arch}.ssm.group_count": "mamba_n_groups",
                    f"{arch}.ssm.time_step_rank": "_ssm_dt_rank",
                    f"{arch}.ssm.inner_size": "_ssm_inner_size",
                    f"{arch}.expert_shared_feed_forward_length": "shared_intermediate_size",
                    f"{arch}.expert_count": "num_local_experts",
                    f"{arch}.expert_used_count": "num_experts_per_tok",
                }
                for gguf_key, cfg_key in extra_fields.items():
                    if gguf_key in reader.fields:
                        field = reader.fields[gguf_key]
                        val = _gguf_parse_value(field.parts[field.data[0]], field.types)
                        config[cfg_key] = val

                # Derive mamba_n_heads and mamba_d_head from inner_size and dt_rank
                # In Mamba2: dt_rank = n_heads; d_head = inner_size / n_heads
                inner = config.pop("_ssm_inner_size", None)
                dt_rank = config.pop("_ssm_dt_rank", None)
                hidden = config.get("hidden_size", 2048)
                if inner and hidden:
                    config["mamba_expand"] = inner // hidden
                if inner and dt_rank:
                    config["mamba_n_heads"] = int(dt_rank)
                    config["mamba_d_head"] = inner // int(dt_rank)

                # Derive layer_types from which layers have SSM tensors
                tensor_names = {t.name for t in reader.tensors}
                num_layers = config.get("num_hidden_layers", 0)
                layer_types = [
                    "mamba" if f"blk.{i}.ssm_in.weight" in tensor_names else "attention"
                    for i in range(num_layers)
                ]
                if layer_types:
                    config["layer_types"] = layer_types

                # Fallback: derive num_key_value_heads from tensor shapes if still 0
                if config.get("num_key_value_heads", 0) == 0:
                    n_heads = config.get("num_attention_heads", 32)
                    head_dim = config.get("hidden_size", 2048) // n_heads
                    for i in range(num_layers):
                        k_name = f"blk.{i}.attn_k.weight"
                        if k_name in tensor_names:
                            for t in reader.tensors:
                                if t.name == k_name:
                                    # GGUF shape is [in, out]; shape[1] = kv_proj_dim
                                    config["num_key_value_heads"] = int(t.shape[1]) // head_dim
                                    break
                            break

            except Exception:
                pass

        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Patch every module that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.tokenization_utils_tokenizers as tok_tokenizers

    for mod in (tok_auto, config_utils, modeling_utils, tok_tokenizers):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 5. Patch get_gguf_hf_weights_map: remap granitemoehybrid → granitehybrid
    #    so gguf-py can look up the tensor name map.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type == "granitemoehybrid":
            model_type = "granitehybrid"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_granitehybrid_gguf()


class ModelVariant(StrEnum):
    """Available Granite 4.0 H-Tiny GGUF model variants for causal language modeling."""

    GRANITE_4_0_H_TINY_GGUF = "H_TINY_GGUF"


class ModelLoader(ForgeModel):
    """Granite 4.0 H-Tiny GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_4_0_H_TINY_GGUF: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-4.0-h-tiny-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_H_TINY_GGUF

    GGUF_FILE = "granite-4.0-h-tiny-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="Granite 4.0 H-Tiny GGUF",
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

        _patch_moe_experts(model)

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

    def load_config(self):
        self._fix_gguf_version_detection()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
