# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski LiquidAI LFM2-24B-A2B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_transformers_lfm2moe_gguf():
    """Monkey-patch transformers to add lfm2moe GGUF architecture support.

    Transformers 5.x has Lfm2MoeForCausalLM but lacks GGUF loading support
    for the lfm2moe architecture. The GGUF file uses 'lfm2moe' as the architecture
    name, but the HF model type is 'lfm2_moe' (with underscore).
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "lfm2moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # Register lfm2moe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")

    # Reuse the lfm2 tensor processor for lfm2moe — it handles the
    # shortconv.conv.weight shape conversion ([hidden, L] -> [hidden, 1, L]).
    from transformers.modeling_gguf_pytorch_utils import TENSOR_PROCESSORS

    if "lfm2moe" not in TENSOR_PROCESSORS and "lfm2" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["lfm2moe"] = TENSOR_PROCESSORS["lfm2"]

    # Add config mapping for lfm2moe (base lfm2 fields + MoE-specific fields)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["lfm2moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "norm_eps",
        "vocab_size": "vocab_size",
        "shortconv.l_cache": "conv_L_cache",
        "expert_count": "num_experts",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_used_count": "num_experts_per_tok",
        "leading_dense_block_count": "num_dense_layers",
        "expert_gating_func": None,
    }

    # Add tokenizer converter (GPT2/BPE-based, same family as lfm2)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )

    if "lfm2moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["lfm2moe"] = GGUFGPTConverter
    if "lfm2_moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["lfm2_moe"] = GGUFGPTConverter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "lfm2moe":
            # Translate GGUF arch name to HF model type
            config["model_type"] = "lfm2_moe"
            # num_key_value_heads is a per-layer list; take the max for HF config
            gguf_kv_heads = config.get("num_key_value_heads", [])
            if isinstance(gguf_kv_heads, list) and gguf_kv_heads:
                full_attn_idxs = [i for i, n in enumerate(gguf_kv_heads) if n > 0]
                config["num_key_value_heads"] = max(gguf_kv_heads)
                config["block_auto_adjust_ff_dim"] = False
                # Lfm2MoeConfig does not auto-compute layer_types; set it explicitly
                n_layers = config.get("num_hidden_layers", len(gguf_kv_heads))
                config["layer_types"] = [
                    "full_attention" if i in full_attn_idxs else "short_conv"
                    for i in range(n_layers)
                ]
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Patch all modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.tokenization_utils_tokenizers as tok_tokenizers

    for mod in (tok_auto, config_utils, modeling_utils, tok_tokenizers):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Patch get_gguf_hf_weights_map to translate lfm2_moe -> lfm2moe for gguf-py lookup.
    # gguf-py's MODEL_ARCH_NAMES uses "lfm2moe" but transformers uses "lfm2_moe".
    orig_weights_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        mt = hf_model.config.model_type if model_type is None else model_type
        if mt == "lfm2_moe":
            mt = "lfm2moe"
        return orig_weights_map(hf_model, processor, mt, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


# Apply the monkey-patch at import time
_patch_transformers_lfm2moe_gguf()

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
    """Available bartowski LiquidAI LFM2-24B-A2B GGUF model variants for causal language modeling."""

    BARTOWSKI_LIQUIDAI_LFM2_24B_A2B_GGUF = "LiquidAI_LFM2_24B_A2B_GGUF"


class ModelLoader(ForgeModel):
    """bartowski LiquidAI LFM2-24B-A2B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_LIQUIDAI_LFM2_24B_A2B_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/LiquidAI_LFM2-24B-A2B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_LIQUIDAI_LFM2_24B_A2B_GGUF

    GGUF_FILE = "LiquidAI_LFM2-24B-A2B-Q4_K_M.gguf"

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
            model="bartowski LiquidAI LFM2-24B-A2B GGUF",
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
        )

        # Tokenizer may have more vocab entries than the GGUF config reports
        # (GGUFGPTConverter adds special tokens). Resize embeddings to match.
        if self.tokenizer is not None and len(self.tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(self.tokenizer))

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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
