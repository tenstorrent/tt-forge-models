# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Solar-Open-100B GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
import os

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


class ModelVariant(StrEnum):
    """Available Solar-Open-100B GGUF model variants for causal language modeling."""

    SOLAR_OPEN_100B_Q4_K_S_GGUF = "Q4_K_S_GGUF"


class ModelLoader(ForgeModel):
    """Solar-Open-100B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SOLAR_OPEN_100B_Q4_K_S_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Solar-Open-100B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SOLAR_OPEN_100B_Q4_K_S_GGUF

    GGUF_FILE = "Solar-Open-100B.Q4_K_S.gguf"

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

    @staticmethod
    def _patch_glm4moe_gguf_support():
        """Register glm4moe GGUF architecture and map it to transformers solar_open model type.

        The Solar-Open-100B GGUF file declares architecture 'glm4moe'. Transformers
        has SolarOpenForCausalLM (model_type='solar_open') which is the correct class.
        This patch bridges the gap.
        """
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils
        import transformers.configuration_utils as _config_utils
        import transformers.models.auto.tokenization_auto as _auto_tokenizer
        import transformers.tokenization_utils_tokenizers as _tok_utils
        from transformers.modeling_gguf_pytorch_utils import (
            GGUF_SUPPORTED_ARCHITECTURES,
        )
        from transformers.integrations.ggml import (
            GGUF_TO_FAST_CONVERTERS,
            GGUFGPTConverter,
        )

        if "glm4moe" in GGUF_SUPPORTED_ARCHITECTURES:
            return

        GGUF_SUPPORTED_ARCHITECTURES.append("glm4moe")

        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["glm4moe"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "embedding_length": "hidden_size",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "expert_used_count": "num_experts_per_tok",
            "expert_count": "n_routed_experts",
            "expert_feed_forward_length": "moe_intermediate_size",
            "expert_shared_count": "n_shared_experts",
            "attention.key_length": "head_dim",
            "vocab_size": "vocab_size",
            "feed_forward_length": None,
            "rope.freq_base": None,
            "rope.dimension_count": None,
            "attention.value_length": None,
            "expert_gating_func": None,
            "expert_weights_scale": None,
            "expert_weights_norm": None,
            "leading_dense_block_count": None,
        }

        if "glm4moe" not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["glm4moe"] = GGUFGPTConverter
        if "solar_open" not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["solar_open"] = GGUFGPTConverter

        _orig_load = _gguf_utils.load_gguf_checkpoint

        def _patched_load_gguf_checkpoint(*args, **kwargs):
            result = _orig_load(*args, **kwargs)
            if result.get("config", {}).get("model_type") == "glm4moe":
                result["config"]["model_type"] = "solar_open"
            return result

        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
            model="Solar-Open-100B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
        self._patch_glm4moe_gguf_support()
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

        if os.environ.get("TT_COMPILE_ONLY_SYSTEM_DESC") or os.environ.get(
            "TT_RANDOM_WEIGHTS"
        ):
            from transformers import SolarOpenForCausalLM

            config = self.config or AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            model = SolarOpenForCausalLM(config)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
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

        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
        self._patch_glm4moe_gguf_support()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
