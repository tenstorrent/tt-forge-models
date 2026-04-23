# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

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


def _patch_transformers_gguf_map():
    # transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING is built once at module
    # import time. Since gguf is installed per-test (not in the base venv), it is absent
    # when the mapping is built. The fallback then reads gguf.__version__ which gguf does
    # not expose, yielding 'N/A' and causing version.parse('N/A') to raise InvalidVersion.
    try:
        import transformers.utils.import_utils as _iu

        if "gguf" not in _iu.PACKAGE_DISTRIBUTION_MAPPING:
            _iu.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
    except Exception:
        pass


def _patch_nemotron_h_moe_support():
    """Register nemotron_h_moe architecture as an alias for nemotron.

    Nemotron-H MoE uses a hybrid SSM+attention+MoE architecture whose GGUF
    type 'nemotron_h_moe' is not yet in transformers. Map it to the standard
    'nemotron' transformer type so that config and tokenizer loading succeeds.
    """
    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "nemotron" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            mapping = dict(
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["nemotron"]
            )
            mapping["expert_count"] = "num_experts"
            mapping["expert_used_count"] = "num_experts_per_tok"
            mapping["expert_feed_forward_length"] = "moe_intermediate_size"
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
                "nemotron_h_moe"
            ] = mapping
    if "nemotron" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUF_TO_FAST_CONVERTERS["nemotron"]
    if "nemotron" in _gguf_utils.TENSOR_PROCESSORS:
        _gguf_utils.TENSOR_PROCESSORS["nemotron_h_moe"] = _gguf_utils.TENSOR_PROCESSORS[
            "nemotron"
        ]


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add nemotron_h_moe support and fix model_type."""
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "nemotron_h_moe":
        result["config"]["model_type"] = "nemotron"
        # NemotronH stores per-layer values as lists; collapse each to a single scalar.
        for key in ("num_key_value_heads", "intermediate_size"):
            val = result["config"].get(key)
            if isinstance(val, list):
                nonzero = [v for v in val if v]
                result["config"][key] = nonzero[0] if nonzero else val[0]
    return result


_patch_nemotron_h_moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Nemotron 3 Super 120B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF = (
        "3_Super_120B_A12B_BF16_heretic_i1_GGUF"
    )
    GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF = "ggml_org_3_Super_120B_GGUF"


class ModelLoader(ForgeModel):
    """Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: LLMModelConfig(
            pretrained_model_name="ggml-org/Nemotron-3-Super-120B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: "NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic.i1-Q4_K_M.gguf",
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: "Nemotron-3-Super-120B-Q4_K.gguf",
    }

    sample_text = "Give me a short introduction to large language model."

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

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
            model="Nemotron 3 Super 120B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_transformers_gguf_map()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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
        _patch_transformers_gguf_map()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
