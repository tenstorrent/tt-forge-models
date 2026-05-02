# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 27B Ultimate Irrefusable Heretic i1 GGUF model loader implementation for causal language modeling.
"""
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


def _patch_transformers_qwen35_gguf():
    """Monkey-patch transformers to add qwen35 GGUF architecture support.

    Qwen3.5 is a hybrid GatedDeltaNet (linear_attn) + full-attention model.
    The GGUF file declares architecture 'qwen35'; transformers needs to load it
    as Qwen3_5TextModel. The SSM metadata (time_step_rank=48 for 27B) must be
    mapped explicitly because the default linear_num_value_heads=32 is wrong.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        GGUFTensor,
        TensorProcessor,
        load_gguf_checkpoint as _orig_load_gguf_checkpoint,
        get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    # 1. Register qwen35 as a supported architecture
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    # 2. Add config mapping for qwen35 (maps to Qwen3_5TextConfig fields)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "ssm.conv_kernel": "linear_conv_kernel_dim",
        "ssm.state_size": "linear_key_head_dim",
        "ssm.time_step_rank": "linear_num_value_heads",
        "ssm.group_count": "linear_num_key_heads",
        "ssm.inner_size": None,
        "full_attention_interval": "full_attention_interval",
        "vocab_size": "vocab_size",
    }

    # 3. Tensor processor for qwen35 SSM-specific weights
    class Qwen35TensorProcessor(TensorProcessor):
        def __init__(self, config=None):
            super().__init__(config=config)

        def preprocess_name(self, hf_name: str) -> str:
            return hf_name.replace(".dt_bias", ".dt_proj")

        def process(self, weights, name, **kwargs):
            if "ssm_conv1d.weight" in name:
                if weights.ndim == 2:
                    weights = np.expand_dims(weights, axis=1)
            if "ssm_a" in name:
                weights = np.log(-weights)
            return GGUFTensor(weights, name, {})

    TENSOR_PROCESSORS["qwen35"] = Qwen35TensorProcessor

    # 4. Register tokenizer converter
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )
    if "qwen35" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen35"] = GGUFQwen2Converter
    if "qwen3_5_text" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3_5_text"] = GGUFQwen2Converter

    # 5. Patch load_gguf_checkpoint: qwen35 → qwen3_5_text + build layer_types
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen35":
            result["config"]["model_type"] = "qwen3_5_text"
            config = result["config"]
            num_layers = config.get("num_hidden_layers", 32)
            interval = config.pop("full_attention_interval", 4)
            layer_types = []
            for i in range(num_layers):
                if (i + 1) % interval == 0:
                    layer_types.append("full_attention")
                else:
                    layer_types.append("linear_attention")
            config["layer_types"] = layer_types
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.tokenization_utils_tokenizers as tok_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, tok_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map: qwen3_5_text → qwen35 for weight lookup
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_text", "qwen3_5"):
            model_type = "qwen35"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


# Apply at module level (runs during test collection)
_patch_transformers_qwen35_gguf()


class ModelVariant(StrEnum):
    """Available Qwen 3.5 27B Ultimate Irrefusable Heretic i1 GGUF model variants for causal language modeling."""

    ULTIMATE_IRREFUSABLE_HERETIC_I1_Q4_K_M = "ultimate_irrefusable_heretic_i1_Q4_K_M"


class ModelLoader(ForgeModel):
    """Qwen 3.5 27B Ultimate Irrefusable Heretic i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.ULTIMATE_IRREFUSABLE_HERETIC_I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-27B-ultimate-irrefusable-heretic-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ULTIMATE_IRREFUSABLE_HERETIC_I1_Q4_K_M

    GGUF_FILE = "Qwen3.5-27B-ultimate-irrefusable-heretic.i1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="Qwen 3.5 27B Ultimate Irrefusable Heretic i1 GGUF",
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

        # Re-apply patch to override any contamination from other loaders
        _patch_transformers_qwen35_gguf()

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

        # Qwen3_5DynamicCache is not a subclass of Cache; disable KV cache
        inputs["use_cache"] = False

        return inputs

    def load_config(self):
        _patch_transformers_qwen35_gguf()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
