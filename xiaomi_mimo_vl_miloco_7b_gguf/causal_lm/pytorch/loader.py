# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
xiaomi-open-source/Xiaomi-MiMo-VL-Miloco-7B-GGUF model loader implementation for causal language modeling.

Loads the Q4_0 GGUF checkpoint via transformers native GGUF support and
extracts the causal LM component from the vision-language model.
"""

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2ForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
)

_prev_load_gguf_checkpoint = _gguf_utils.load_gguf_checkpoint
_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map

# LLM config keys to route into text_config when constructing Qwen2_5_VLConfig
_QWEN25VL_TEXT_CONFIG_KEYS = {
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
}


def _patch_qwen2vl_support():
    """Register qwen2vl in transformers GGUF tables, using qwen2 mappings for the LLM part."""
    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    for section in GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen2" in GGUF_TO_TRANSFORMERS_MAPPING[section]:
            GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen2vl",
                GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen2"],
            )
    if "qwen2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen2vl", GGUF_TO_FAST_CONVERTERS["qwen2"])


def _find_real_load_gguf_checkpoint(fn):
    """Walk the monkey-patch chain to find the real transformers function.

    Walks via module globals (_orig_load_gguf_checkpoint, _prev_load_gguf_checkpoint)
    and also via closure variables named 'orig_load' (used by some loaders such as
    glm_4_32b that define their captured reference as a local inside a helper function).
    """
    seen = set()
    while fn is not None:
        fid = id(fn)
        if fid in seen:
            break
        seen.add(fid)
        if getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils":
            return fn
        globs = getattr(fn, "__globals__", {})
        nxt = globs.get("_orig_load_gguf_checkpoint") or globs.get(
            "_prev_load_gguf_checkpoint"
        )
        if not callable(nxt):
            # Some loaders capture the original as a closure variable 'orig_load'
            # rather than a module-level global; walk into those closures too.
            code = getattr(fn, "__code__", None)
            closure = getattr(fn, "__closure__", None) or ()
            if code and closure:
                for varname, cell in zip(code.co_freevars, closure):
                    if varname == "orig_load":
                        try:
                            val = cell.cell_contents
                            if callable(val) and id(val) not in seen:
                                nxt = val
                                break
                        except ValueError:
                            pass
        if not callable(nxt):
            break
        fn = nxt
    return fn


_real_load_gguf_checkpoint = _find_real_load_gguf_checkpoint(_prev_load_gguf_checkpoint)


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to handle qwen2_5_vl model type."""
    if model_type is None and hasattr(hf_model, "config"):
        model_type = hf_model.config.model_type
    if model_type == "qwen2_5_vl":
        model_type = "qwen2vl"
        # Get num_layers from text_config since top-level config doesn't expose it
        if num_layers is None:
            cfg = getattr(hf_model, "config", None)
            if cfg is not None:
                text_cfg = getattr(cfg, "text_config", None)
                if text_cfg is not None:
                    num_layers = getattr(text_cfg, "num_hidden_layers", None)
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen2vl support, accepting transformers 5.x kwargs."""
    _patch_qwen2vl_support()
    result = _real_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    config = result.get("config", {})
    if config.get("model_type") == "qwen2vl":
        # Restructure flat LLM params into the nested text_config expected by Qwen2_5_VLConfig
        text_config = {
            k: config.pop(k) for k in list(config) if k in _QWEN25VL_TEXT_CONFIG_KEYS
        }
        config["model_type"] = "qwen2_5_vl"
        config["text_config"] = text_config
    result["config"] = config
    return result


_patch_qwen2vl_support()
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


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
    """Available Xiaomi-MiMo-VL-Miloco-7B-GGUF model variants for causal language modeling."""

    XIAOMI_MIMO_VL_MILOCO_7B_GGUF = "7B_GGUF"


class ModelLoader(ForgeModel):
    """Xiaomi-MiMo-VL-Miloco-7B-GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.XIAOMI_MIMO_VL_MILOCO_7B_GGUF: LLMModelConfig(
            pretrained_model_name="xiaomi-open-source/Xiaomi-MiMo-VL-Miloco-7B-GGUF",
            max_length=128,
        ),
    }

    GGUF_FILE = "MiMo-VL-Miloco-7B_Q4_0.gguf"
    BASE_MODEL = "xiaomi-open-source/Xiaomi-MiMo-VL-Miloco-7B"

    DEFAULT_VARIANT = ModelVariant.XIAOMI_MIMO_VL_MILOCO_7B_GGUF

    sample_text = "Describe the key features of a vision-language model."

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
            model="xiaomi-open-source Xiaomi-MiMo-VL-Miloco-7B-GGUF",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Reinstall patches at load time to override any broken patch chains from
        # alphabetically-later loaders imported before this one.
        _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        # GGUF contains only LLM weights; vision encoder weights are absent/mismatched
        model_kwargs.setdefault("ignore_mismatched_sizes", True)
        model_kwargs |= kwargs

        # The GGUF repo has no config.json, so supply it from the base model.
        # Use /tmp cache to avoid filling the data disk with the 4.4 GB GGUF.
        vl_config = AutoConfig.from_pretrained(self.BASE_MODEL)

        # Load the full VL model from GGUF, then extract the causal LM component.
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=vl_config,
            cache_dir="/tmp/hf_cache",
            **model_kwargs
        )
        text_config = full_model.config.text_config
        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers
        model = Qwen2ForCausalLM(text_config)
        model.model = full_model.model.language_model
        model.lm_head = full_model.lm_head
        model.eval()

        self.config = text_config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
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
        config = AutoConfig.from_pretrained(self.BASE_MODEL)
        self.config = config.text_config
        return self.config
