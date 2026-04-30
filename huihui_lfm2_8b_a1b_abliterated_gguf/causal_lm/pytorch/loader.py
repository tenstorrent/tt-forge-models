# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui LFM2 8B A1B Abliterated GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_lfm2moe_gguf():
    """Register the lfm2moe GGUF architecture in the transformers GGUF loading pipeline.

    transformers 5.2.0 knows about lfm2_moe as an HF model type but lacks the
    GGUF_CONFIG_MAPPING entry for the raw GGUF architecture key "lfm2moe".
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        Lfm2TensorProcessor,
        TENSOR_PROCESSORS,
    )

    if "lfm2moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register lfm2moe in the GGUF config mapping.
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
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
        "leading_dense_block_count": "num_dense_layers",
    }

    # 2. Update the derived supported-architectures list.
    GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")

    # 3. Use the same tensor processor as lfm2 (expands shortconv.conv.weight dims).
    TENSOR_PROCESSORS["lfm2moe"] = Lfm2TensorProcessor

    # 4. Register the BPE tokenizer converter (same as qwen2 / deepseek2).
    #    tokenization_utils_tokenizers.py looks up by model_type (i.e. "lfm2_moe"
    #    after our remap), so we must register under the remapped key.
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter
    for _key in ("lfm2moe", "lfm2_moe"):
        if _key not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS[_key] = GGUFQwen2Converter

    # 5. Patch load_gguf_checkpoint to:
    #    a) Convert per-layer num_key_value_heads list to a scalar (max value).
    #    b) Build layer_types list from non-zero kv-head indices (Lfm2MoeConfig
    #       needs layer_types, not full_attn_idxs which only Lfm2Config accepts).
    #    c) Remap model_type "lfm2moe" → "lfm2_moe" so AutoConfig finds Lfm2MoeConfig.
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "lfm2moe":
            kv_heads = config.get("num_key_value_heads")
            if isinstance(kv_heads, list):
                config["layer_types"] = [
                    "full_attention" if n > 0 else "conv" for n in kv_heads
                ]
                config["num_key_value_heads"] = max(kv_heads)
            config["model_type"] = "lfm2_moe"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Patch every module that imported load_gguf_checkpoint at module level.
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.models.auto.tokenization_auto as tok_auto

    for mod in (config_utils, modeling_utils, tok_auto):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map to remap "lfm2_moe" → "lfm2moe" so the
    #    gguf-py tensor-name map lookup resolves MODEL_ARCH.LFM2MOE correctly.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        if model_type == "lfm2_moe" or (
            model_type is None
            and getattr(getattr(hf_model, "config", None), "model_type", None) == "lfm2_moe"
        ):
            model_type = "lfm2moe"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_lfm2moe_gguf()


class ModelVariant(StrEnum):
    """Available Huihui LFM2 8B A1B Abliterated GGUF model variants for causal language modeling."""

    HUIHUI_LFM2_8B_A1B_ABLITERATED_GGUF = "8B_A1B_Abliterated_GGUF"


class ModelLoader(ForgeModel):
    """Huihui LFM2 8B A1B Abliterated GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_LFM2_8B_A1B_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-LFM2-8B-A1B-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_LFM2_8B_A1B_ABLITERATED_GGUF

    GGUF_FILE = "Huihui-LFM2-8B-A1B-abliterated.Q4_K_M.gguf"

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
            model="Huihui LFM2 8B A1B Abliterated GGUF",
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
