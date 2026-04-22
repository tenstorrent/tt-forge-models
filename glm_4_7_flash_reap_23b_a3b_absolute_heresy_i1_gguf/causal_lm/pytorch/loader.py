# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM 4.7 Flash REAP 23B A3B absolute heresy i1 GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_deepseek2_gguf():
    """Register deepseek2/deepseek_v2 GGUF architecture and tokenizer support.

    This model's GGUF file uses 'deepseek_v2' as the tokenizer class name and
    'deepseek2' as the model architecture, neither of which transformers 5.x
    recognises out of the box.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "deepseek2" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["deepseek2"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "rope.dimension_count": "qk_rope_head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": None,
        "attention.value_length": None,
        "attention.key_length_mla": "qk_nope_head_dim",
        "attention.value_length_mla": "v_head_dim",
        "attention.q_lora_rank": "q_lora_rank",
        "attention.kv_lora_rank": "kv_lora_rank",
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_count": "n_shared_experts",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_weights_scale": "routed_scaling_factor",
        "expert_weights_norm": "norm_topk_prob",
        "leading_dense_block_count": "first_k_dense_replace",
        "expert_feed_forward_length": "moe_intermediate_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    GGUF_TO_FAST_CONVERTERS.setdefault("deepseek2", GGUFQwen2Converter)
    GGUF_TO_FAST_CONVERTERS.setdefault("deepseek_v2", GGUFQwen2Converter)

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "deepseek2":
            config["model_type"] = "deepseek_v2"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.tokenization_utils_tokenizers as tok_utils

    for mod in (tok_auto, config_utils, modeling_utils, tok_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_deepseek2_gguf()


class ModelVariant(StrEnum):
    """Available GLM 4.7 Flash REAP 23B A3B absolute heresy i1 GGUF model variants for causal language modeling."""

    GLM_4_7_FLASH_REAP_23B_A3B_ABSOLUTE_HERESY_I1_GGUF = (
        "4_7_Flash_REAP_23B_A3B_absolute_heresy_i1_GGUF"
    )


class ModelLoader(ForgeModel):
    """GLM 4.7 Flash REAP 23B A3B absolute heresy i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_REAP_23B_A3B_ABSOLUTE_HERESY_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/GLM-4.7-Flash-REAP-23B-A3B-absolute-heresy-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_REAP_23B_A3B_ABSOLUTE_HERESY_I1_GGUF

    GGUF_FILE = "GLM-4.7-Flash-REAP-23B-A3B-absolute-heresy.i1-Q4_K_M.gguf"

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
            model="GLM 4.7 Flash REAP 23B A3B absolute heresy i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        # Fix stale PACKAGE_DISTRIBUTION_MAPPING when gguf is installed after transformers import
        import gguf as _gguf
        import importlib.metadata
        import transformers.utils.import_utils as _iu

        if not hasattr(_gguf, "__version__"):
            try:
                _gguf.__version__ = importlib.metadata.version("gguf")
            except Exception:
                pass
        _iu.PACKAGE_DISTRIBUTION_MAPPING.setdefault("gguf", ["gguf"])

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
