# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui LFM2 24B A2B Abliterated i1-GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
    TENSOR_PROCESSORS,
    Lfm2TensorProcessor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

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


def _patch_lfm2moe_support():
    """Register lfm2moe GGUF architecture as an alias for the lfm2_moe transformers model.

    The GGUF file for LFM2 MoE models declares architecture as 'lfm2moe', but
    transformers only recognises 'lfm2'. The MoE variant has a dedicated
    Lfm2MoeForCausalLM class whose model_type is 'lfm2_moe'.
    """
    if "lfm2moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")
    for section, mapping in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING.items():
        if "lfm2" in mapping:
            mapping.setdefault(
                "lfm2moe",
                {
                    **mapping["lfm2"],
                    "expert_count": "num_experts",
                    "expert_used_count": "num_experts_per_tok",
                    "expert_feed_forward_length": "moe_intermediate_size",
                    "leading_dense_block_count": "num_dense_layers",
                },
            )
    TENSOR_PROCESSORS.setdefault("lfm2moe", Lfm2TensorProcessor)
    # lfm2moe uses a gpt2-style BPE tokenizer
    GGUF_TO_FAST_CONVERTERS.setdefault("lfm2moe", GGUFGPTConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("lfm2_moe", GGUFGPTConverter)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    """Wrap load_gguf_checkpoint to add lfm2moe support and fix model_type."""
    _patch_lfm2moe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
    )
    if result.get("config", {}).get("model_type") == "lfm2moe":
        # Apply same num_key_value_heads fix as for lfm2 (list -> max scalar)
        gguf_num_kv = result["config"].get("num_key_value_heads")
        if isinstance(gguf_num_kv, list):
            result["config"]["num_key_value_heads"] = max(gguf_num_kv)
            result["config"]["block_auto_adjust_ff_dim"] = False
            result["config"]["full_attn_idxs"] = [
                i for i, n in enumerate(gguf_num_kv) if n > 0
            ]
            # Lfm2MoeConfig uses layer_types (list of "full_attention"/"short_conv")
            result["config"]["layer_types"] = [
                "full_attention" if n > 0 else "short_conv" for n in gguf_num_kv
            ]
        result["config"]["model_type"] = "lfm2_moe"
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to map lfm2_moe back to the GGUF lfm2moe arch name."""
    resolved_model_type = (
        hf_model.config.model_type if model_type is None else model_type
    )
    if resolved_model_type == "lfm2_moe":
        model_type = "lfm2moe"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


_patch_lfm2moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Huihui LFM2 24B A2B Abliterated i1-GGUF model variants for causal language modeling."""

    HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF = "24B_A2B_i1_GGUF"
    HUIHUI_LFM2_24B_A2B_ABLITERATED_GGUF = "24B_A2B_GGUF"


class ModelLoader(ForgeModel):
    """Huihui LFM2 24B A2B Abliterated i1-GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-LFM2-24B-A2B-abliterated-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-LFM2-24B-A2B-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF: "Huihui-LFM2-24B-A2B-abliterated.i1-Q4_K_M.gguf",
        ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_GGUF: "Huihui-LFM2-24B-A2B-abliterated.Q4_K_M.gguf",
    }

    sample_text = "The quick brown fox jumps over the lazy dog."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Huihui LFM2 24B A2B Abliterated i1-GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        # GGUFGPTConverter adds <s>/<</s> as added tokens with out-of-range IDs (65536/65537).
        # Override with the correct in-vocabulary special tokens from the GGUF metadata.
        self.tokenizer.bos_token = "<|startoftext|>"
        self.tokenizer.eos_token = "<|im_end|>"
        self.tokenizer.pad_token = "<|pad|>"

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
