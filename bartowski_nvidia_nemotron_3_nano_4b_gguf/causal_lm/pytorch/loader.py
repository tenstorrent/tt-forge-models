# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski nvidia Nemotron-3-Nano-4B GGUF model loader implementation for causal language modeling.
"""
import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# nemotron_h (Nemotron-3-Nano hybrid Mamba+Attention) is not yet in transformers
# GGUF supported architectures list. This patch adds the necessary config-key
# mapping and post-loading fixups so that AutoModelForCausalLM.from_pretrained
# with gguf_file= can load these checkpoints.

_NEMOTRON_H_CONFIG_MAP = {
    "block_count": "num_hidden_layers",
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "vocab_size": "vocab_size",
    "attention.head_count": "num_attention_heads",
    # head_count_kv is 0 in GGUF (not per-layer); skip and rely on
    # num_key_value_heads derived from tensor shapes later.
    "attention.head_count_kv": -1,
    "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
    # key_length gives the per-head dimension for attention layers.
    "attention.key_length": "head_dim",
    "attention.value_length": -1,
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "rope.scaling.finetuned": -1,
    # SSM parameters
    "ssm.conv_kernel": "conv_kernel",
    "ssm.state_size": "ssm_state_size",
    "ssm.group_count": "n_groups",
    # ssm.time_step_rank corresponds to mamba_num_heads in NemotronHConfig
    "ssm.time_step_rank": "mamba_num_heads",
    # ssm.inner_size = mamba_num_heads * mamba_head_dim; handled in post-load
    "ssm.inner_size": None,
}


def _infer_layers_block_type(reader) -> list:
    """Derive layers_block_type from which tensors each block contains."""
    layer_tensor_sets = {}
    for t in reader.tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m:
            idx = int(m.group(1))
            layer_tensor_sets.setdefault(idx, set()).add(t.name.split(".")[2])

    if not layer_tensor_sets:
        return []
    num_layers = max(layer_tensor_sets.keys()) + 1
    block_types = []
    for i in range(num_layers):
        keys = layer_tensor_sets.get(i, set())
        if "ssm_a" in keys or "ssm_in" in keys:
            block_types.append("mamba")
        elif "attn_q" in keys or "attn_k" in keys:
            block_types.append("attention")
        else:
            block_types.append("mlp")
    return block_types


def _patch_nemotron_h_support():
    if "nemotron_h" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "nemotron_h"
    ] = _NEMOTRON_H_CONFIG_MAP
    # Reuse the existing nemotron tokenizer converter for nemotron_h.
    if (
        "nemotron_h" not in GGUF_TO_FAST_CONVERTERS
        and "nemotron" in GGUF_TO_FAST_CONVERTERS
    ):
        GGUF_TO_FAST_CONVERTERS["nemotron_h"] = GGUF_TO_FAST_CONVERTERS["nemotron"]


def _patched_load_gguf_checkpoint(
    gguf_path, return_tensors=False, model_to_load=None, torch_dtype=None
):
    _patch_nemotron_h_support()

    # Other loaders in the chain may not accept torch_dtype; only pass it when
    # set to avoid TypeError from their incompatible function signatures.
    extra_kw = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
    result = _orig_load_gguf_checkpoint(
        gguf_path,
        return_tensors=return_tensors,
        model_to_load=model_to_load,
        **extra_kw
    )

    if result.get("config", {}).get("model_type") == "nemotron_h":
        cfg = result["config"]

        # layers_block_type must be inferred from tensor presence per block.
        if "layers_block_type" not in cfg:
            try:
                from gguf import GGUFReader

                _reader = GGUFReader(gguf_path)
                cfg["layers_block_type"] = _infer_layers_block_type(_reader)
            except Exception:
                pass

        # For hybrid models the GGUF loader may produce per-layer lists for
        # fields like feed_forward_length and attention.head_count when those
        # values differ across layer types. NemotronHConfig expects scalars, so
        # collapse any list by taking the max of its non-zero elements.
        for key in ("intermediate_size", "num_attention_heads"):
            val = cfg.get(key)
            if isinstance(val, list):
                non_zero = [v for v in val if v]
                cfg[key] = int(max(non_zero)) if non_zero else 0

        # Ensure GGUF-derived values are Python ints (GGUF shapes are np.uint64).
        for key in (
            "mamba_num_heads",
            "head_dim",
            "num_attention_heads",
            "intermediate_size",
        ):
            if key in cfg and not isinstance(cfg[key], int):
                cfg[key] = int(cfg[key])

        # Derive mamba_head_dim from ssm.inner_size and mamba_num_heads.
        try:
            from gguf import GGUFReader

            _reader = GGUFReader(gguf_path)
            inner_size_field = _reader.fields.get("nemotron_h.ssm.inner_size")
            if inner_size_field is not None:
                inner_size = int(inner_size_field.parts[-1][0])
                mamba_num_heads = int(cfg.get("mamba_num_heads", 96))
                if mamba_num_heads > 0:
                    cfg["mamba_head_dim"] = int(inner_size // mamba_num_heads)
        except Exception:
            pass

        # GGUF stores head_count_kv=0 for hybrid models. Infer from K-proj shape.
        if cfg.get("num_key_value_heads", 0) == 0:
            try:
                from gguf import GGUFReader

                _reader = GGUFReader(gguf_path)
                head_dim = int(cfg.get("head_dim", 128))
                for t in _reader.tensors:
                    if re.match(r"blk\.\d+\.attn_k\.weight", t.name):
                        k_out_dim = (
                            int(t.shape[1]) if len(t.shape) >= 2 else int(t.shape[0])
                        )
                        cfg["num_key_value_heads"] = int(k_out_dim // head_dim)
                        break
            except Exception:
                cfg["num_key_value_heads"] = 8

        cfg.setdefault("num_key_value_heads", 8)

    return result


_patch_nemotron_h_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available bartowski nvidia Nemotron-3-Nano-4B GGUF model variants for causal language modeling."""

    BARTOWSKI_NVIDIA_NEMOTRON_3_NANO_4B_Q4_K_M_GGUF = (
        "nvidia_Nemotron_3_Nano_4B_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """bartowski nvidia Nemotron-3-Nano-4B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_NVIDIA_NEMOTRON_3_NANO_4B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/nvidia_Nemotron-3-Nano-4B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_NVIDIA_NEMOTRON_3_NANO_4B_Q4_K_M_GGUF

    GGUF_FILE = "nvidia_Nemotron-3-Nano-4B-Q4_K_M.gguf"

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
            model="bartowski nvidia Nemotron-3-Nano-4B GGUF",
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
