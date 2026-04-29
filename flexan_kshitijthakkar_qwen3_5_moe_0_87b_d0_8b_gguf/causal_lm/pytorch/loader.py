# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flexan Kshitijthakkar Qwen3.5 MoE 0.87B d0.8B GGUF model loader implementation for causal language modeling.
"""
import contextlib
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


def _make_qwen35moe_tensor_processor():
    """Build and return a TensorProcessor subclass for qwen35moe.

    qwen35moe is an SSM/Mamba-attention hybrid with MoE FFN.  Two special
    cases beyond the standard Qwen2MoeTensorProcessor:

    1. ssm_conv1d.weight is stored as [C, kernel] in GGUF but HF Conv1d
       (grouped) expects [C, 1, kernel] — we insert the groups dimension.
    2. Expert gate/up weights are stored separately (ffn_gate_exps /
       ffn_up_exps) rather than pre-merged, so tensor_key_mapping entries
       injected by patched_get_gguf_hf_weights_map resolve to the same
       gate_up_proj HF tensor, and _set_moe_expert_tensor handles
       interleaving (inherited from Qwen2MoeTensorProcessor).
    """
    import numpy as np
    from transformers.modeling_gguf_pytorch_utils import Qwen2MoeTensorProcessor

    class Qwen35MoeTensorProcessor(Qwen2MoeTensorProcessor):
        def process(self, weights, name: str, **kwargs):
            # Expand ssm_conv1d from [C, kernel] → [C, 1, kernel]
            if "ssm_conv1d.weight" in name:
                weights = np.expand_dims(weights, axis=1)
            return super().process(weights, name, **kwargs)

    return Qwen35MoeTensorProcessor


def _setup_qwen35moe_arch():
    """Register qwen35moe architecture mappings in transformers (idempotent)."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen35moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "full_attention_interval": "full_attention_interval",
        # MoE feed-forward sizes
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
        # SSM / linear-attention parameters
        # ssm.time_step_rank → linear_num_key_heads (and copied to linear_num_value_heads
        # in patched_load_gguf_checkpoint since there is no 1-to-2 mapping support)
        "ssm.time_step_rank": "linear_num_key_heads",
        "ssm.state_size": "linear_key_head_dim",
        "ssm.conv_kernel": "linear_conv_kernel_dim",
        "ssm.inner_size": None,  # stored only for reference; derived sizes handled below
    }

    # Always install our custom processor (handles both MoE expert tensors and
    # SSM conv1d weight shape expansion).
    TENSOR_PROCESSORS["qwen35moe"] = _make_qwen35moe_tensor_processor()

    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen35moe", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        )
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_moe_text", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        )


def _get_real_load_gguf_checkpoint():
    """Return the original transformers load_gguf_checkpoint.

    In a full pytest session, many loaders install broken patches of
    load_gguf_checkpoint that drop the model_to_load kwarg added in
    transformers 5.x.  These form a CHAIN:

      - Module-level patchers store the previous function in their module
        globals as _orig_load_gguf_checkpoint.
      - Function-level patchers (e.g. noctrex) store it as a closure variable
        (orig_load / _orig).

    We follow this chain linearly until we find a function defined in
    transformers.modeling_gguf_pytorch_utils with qualname load_gguf_checkpoint
    (the original).  That function accepts model_to_load.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    def _is_real(fn):
        return (
            getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils"
            and getattr(fn, "__qualname__", "") == "load_gguf_checkpoint"
        )

    fn = gguf_utils.load_gguf_checkpoint
    seen_ids = set()

    while not _is_real(fn):
        fn_id = id(fn)
        if fn_id in seen_ids:
            # Cycle detected; give up and return whatever we have.
            break
        seen_ids.add(fn_id)

        next_fn = None

        # 1. Module-level patchers: _orig_load_gguf_checkpoint in globals
        orig = fn.__globals__.get("_orig_load_gguf_checkpoint")
        if orig is not None and callable(orig) and id(orig) not in seen_ids:
            next_fn = orig

        # 2. Function-level patchers: look for gguf-related callables in closure
        if next_fn is None:
            for cell in getattr(fn, "__closure__", None) or []:
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if (
                    callable(val)
                    and id(val) not in seen_ids
                    and "gguf" in getattr(val, "__name__", "").lower()
                ):
                    next_fn = val
                    break

        # 3. Fallback: scan globals for any load_gguf-named callable not yet seen
        if next_fn is None:
            for _name, val in list(fn.__globals__.items()):
                if (
                    callable(val)
                    and id(val) not in seen_ids
                    and "load_gguf" in getattr(val, "__name__", "").lower()
                ):
                    next_fn = val
                    break

        if next_fn is None:
            break  # No more leads in this link

        fn = next_fn

    # If we found the real function, return it; otherwise fall back to whatever
    # is current so the test fails with the original error rather than a new one.
    if _is_real(fn):
        return fn
    return gguf_utils.load_gguf_checkpoint


@contextlib.contextmanager
def _qwen35moe_gguf_context():
    """Context manager that installs a correct load_gguf_checkpoint for this model.

    Finds the REAL transformers function (bypassing any broken chain), wraps it
    to remap qwen35moe→qwen3_5_moe_text and generate layer_types, then patches
    ALL binding sites so modeling_utils.py line 4010 gets the correct function
    regardless of test-session import order.

    On exit, restores all binding sites to whatever they were before.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    try:
        import transformers.tokenization_utils_tokenizers as tok_utils_mod
        _has_tok_utils = True
    except ImportError:
        _has_tok_utils = False

    real_load = _get_real_load_gguf_checkpoint()

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = real_load(*args, **kwargs)
        cfg = result.get("config", {})
        if cfg.get("model_type") == "qwen35moe":
            cfg["model_type"] = "qwen3_5_moe_text"
            num_layers = cfg.get("num_hidden_layers", 24)
            interval = cfg.pop("full_attention_interval", 4)
            layer_types = []
            for i in range(num_layers):
                layer_types.append(
                    "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                )
            cfg["layer_types"] = layer_types
            # ssm.time_step_rank → both linear_num_key_heads and linear_num_value_heads
            # (GGUF mapping only supports 1:1 so we mirror the key heads value)
            if "linear_num_key_heads" in cfg:
                cfg.setdefault("linear_num_value_heads", cfg["linear_num_key_heads"])
            # ssm.state_size → both linear_key_head_dim and linear_value_head_dim
            if "linear_key_head_dim" in cfg:
                cfg.setdefault("linear_value_head_dim", cfg["linear_key_head_dim"])
        return result

    # Patch get_gguf_hf_weights_map to map qwen3_5_moe_text back to qwen35moe
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
            model_type = "qwen35moe"
        result = orig_get_map(hf_model, processor, model_type, num_layers, qual_name)
        # qwen35moe GGUF files store gate/up expert weights separately as
        # blk.N.ffn_gate_exps and blk.N.ffn_up_exps, but the gguf-py arch
        # name_map maps gate_up_proj → ffn_gate_up_exps (merged form).
        # We inject both split-tensor keys pointing to the same merged HF
        # weight so that Qwen2MoeTensorProcessor.process() can find them.
        import re as _re
        _gate_up_pat = _re.compile(
            r"blk\.(\d+)\.ffn_gate_up_exps(\.weight)?$"
        )
        extra = {}
        for gguf_key, hf_val in result.items():
            m = _gate_up_pat.fullmatch(gguf_key)
            if m:
                bid = m.group(1)
                sfx = m.group(2) or ""
                extra[f"blk.{bid}.ffn_gate_exps{sfx}"] = hf_val
                extra[f"blk.{bid}.ffn_up_exps{sfx}"] = hf_val
        result.update(extra)
        return result

    # Save and replace all binding sites
    saved = {
        "gguf_utils": gguf_utils.load_gguf_checkpoint,
        "tok_auto": getattr(tok_auto, "load_gguf_checkpoint", None),
        "config_utils": getattr(config_utils, "load_gguf_checkpoint", None),
        "modeling_utils": getattr(modeling_utils, "load_gguf_checkpoint", None),
        "get_map": gguf_utils.get_gguf_hf_weights_map,
    }

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map
    if saved["tok_auto"] is not None:
        tok_auto.load_gguf_checkpoint = patched_load_gguf_checkpoint
    if saved["config_utils"] is not None:
        config_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    if saved["modeling_utils"] is not None:
        modeling_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    if _has_tok_utils:
        import transformers.tokenization_utils_tokenizers as tok_utils_mod2
        if hasattr(tok_utils_mod2, "load_gguf_checkpoint"):
            saved["tok_utils"] = tok_utils_mod2.load_gguf_checkpoint
            tok_utils_mod2.load_gguf_checkpoint = patched_load_gguf_checkpoint

    try:
        yield
    finally:
        # Restore all binding sites
        gguf_utils.load_gguf_checkpoint = saved["gguf_utils"]
        gguf_utils.get_gguf_hf_weights_map = saved["get_map"]
        if saved["tok_auto"] is not None:
            tok_auto.load_gguf_checkpoint = saved["tok_auto"]
        if saved["config_utils"] is not None:
            config_utils.load_gguf_checkpoint = saved["config_utils"]
        if saved["modeling_utils"] is not None:
            modeling_utils.load_gguf_checkpoint = saved["modeling_utils"]
        if _has_tok_utils and "tok_utils" in saved:
            import transformers.tokenization_utils_tokenizers as tok_utils_mod2
            tok_utils_mod2.load_gguf_checkpoint = saved["tok_utils"]


# Register architecture mappings at import time (idempotent metadata only,
# no load_gguf_checkpoint patching here to avoid race conditions with other loaders)
_setup_qwen35moe_arch()


class ModelVariant(StrEnum):
    """Available Flexan Kshitijthakkar Qwen3.5 MoE 0.87B d0.8B GGUF model variants for causal language modeling."""

    FLEXAN_KSHITIJTHAKKAR_QWEN3_5_MOE_0_87B_D0_8B_GGUF = "MoE_0.87B_d0.8B_GGUF"


class ModelLoader(ForgeModel):
    """Flexan Kshitijthakkar Qwen3.5 MoE 0.87B d0.8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FLEXAN_KSHITIJTHAKKAR_QWEN3_5_MOE_0_87B_D0_8B_GGUF: LLMModelConfig(
            pretrained_model_name="Flexan/kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLEXAN_KSHITIJTHAKKAR_QWEN3_5_MOE_0_87B_D0_8B_GGUF

    GGUF_FILE = "qwen3.5-moe-0.87B-d0.8B.Q4_K_M.gguf"

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
            model="Flexan Kshitijthakkar Qwen3.5 MoE 0.87B d0.8B GGUF",
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

        with _qwen35moe_gguf_context():
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
            with _qwen35moe_gguf_context():
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
                if hasattr(config, "layer_types"):
                    config.layer_types = config.layer_types[: self.num_layers]
            model_kwargs["config"] = config

        with _qwen35moe_gguf_context():
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

    def load_config(self):
        with _qwen35moe_gguf_context():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
