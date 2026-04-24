# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher HY-MT1.5-1.8B i1 GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES

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


def _patch_hunyuan_dense_support():
    """Register hunyuan-dense arch in GGUF loaders and AutoConfig mapping."""
    import transformers.integrations.ggml as _ggml
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.hunyuan_v1_dense.configuration_hunyuan_v1_dense import (
        HunYuanDenseV1Config,
    )

    if "hunyuan-dense" not in GGUF_SUPPORTED_ARCHITECTURES:
        _ggml.GGUF_CONFIG_MAPPING["hunyuan-dense"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "attention.key_length": "head_dim",
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "vocab_size": "vocab_size",
        }
        GGUF_SUPPORTED_ARCHITECTURES.append("hunyuan-dense")

    if "hunyuan-dense" not in _ggml.GGUF_TO_FAST_CONVERTERS:
        _ggml.GGUF_TO_FAST_CONVERTERS["hunyuan-dense"] = _ggml.GGUFGPTConverter

    try:
        CONFIG_MAPPING.register("hunyuan-dense", HunYuanDenseV1Config, exist_ok=True)
    except Exception:
        pass


def _apply_gguf_compat_patches():
    """Patch load_gguf_checkpoint and get_gguf_hf_weights_map to accept/handle
    newer transformers kwargs (e.g. model_to_load) that older monkey-patches
    in the test session drop silently, breaking the call chain."""
    import inspect
    import transformers.modeling_gguf_pytorch_utils as _gguf_mod

    chain_top = _gguf_mod.load_gguf_checkpoint
    try:
        needs_compat = "model_to_load" not in inspect.signature(chain_top).parameters
    except Exception:
        needs_compat = False

    if not needs_compat:
        return

    def _compat_load_gguf(gguf_path, return_tensors=False, **kw):
        kw.pop("model_to_load", None)
        return chain_top(gguf_path, return_tensors=return_tensors, **kw)

    orig_get_map = _gguf_mod.get_gguf_hf_weights_map

    def _compat_get_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if hf_model is None:
            return {}
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_mod.load_gguf_checkpoint = _compat_load_gguf
    _gguf_mod.get_gguf_hf_weights_map = _compat_get_map


class ModelVariant(StrEnum):
    """Available mradermacher HY-MT1.5-1.8B i1 GGUF model variants for causal language modeling."""

    HY_MT1_5_1_8B_I1_GGUF = "1.8B_i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher HY-MT1.5-1.8B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HY_MT1_5_1_8B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/HY-MT1.5-1.8B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HY_MT1_5_1_8B_I1_GGUF

    GGUF_FILE = "HY-MT1.5-1.8B.i1-Q4_K_M.gguf"

    sample_text = (
        "Translate the following segment into Chinese: The weather is nice today."
    )

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
        self._fix_gguf_version_detection()

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher HY-MT1.5-1.8B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_hunyuan_dense_support()

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
        self._fix_gguf_version_detection()
        _patch_hunyuan_dense_support()
        _apply_gguf_compat_patches()

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
        _patch_hunyuan_dense_support()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
