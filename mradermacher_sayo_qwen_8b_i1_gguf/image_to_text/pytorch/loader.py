# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Sayo-Qwen-8B i1 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
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

_prev_load_gguf_checkpoint = _gguf_utils.load_gguf_checkpoint
_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map

# LLM config keys to route into text_config when constructing Qwen3VLConfig
_QWEN3VL_TEXT_CONFIG_KEYS = {
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


def _patch_qwen3vl_support():
    """Register qwen3vl in transformers GGUF tables, using qwen3 mappings for the LLM part."""
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    for section in GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in GGUF_TO_TRANSFORMERS_MAPPING[section]:
            GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen3vl",
                GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUF_TO_FAST_CONVERTERS["qwen3"])


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
    """Wrap get_gguf_hf_weights_map to handle qwen3_vl model type."""
    if model_type is None and hasattr(hf_model, "config"):
        model_type = hf_model.config.model_type
    if model_type == "qwen3_vl":
        model_type = "qwen3vl"
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
    """Wrap load_gguf_checkpoint to add qwen3vl support, accepting transformers 5.x kwargs."""
    _patch_qwen3vl_support()
    result = _real_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    config = result.get("config", {})
    if config.get("model_type") == "qwen3vl":
        # Restructure flat LLM params into the nested text_config expected by Qwen3VLConfig
        text_config = {
            k: config.pop(k) for k in list(config) if k in _QWEN3VL_TEXT_CONFIG_KEYS
        }
        config["model_type"] = "qwen3_vl"
        config["text_config"] = text_config
    result["config"] = config
    return result


_patch_qwen3vl_support()
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available mradermacher Sayo-Qwen-8B i1 GGUF model variants for image to text."""

    SAYO_QWEN_8B_I1_Q4_K_M_GGUF = "8B_I1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher Sayo-Qwen-8B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.SAYO_QWEN_8B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Sayo-Qwen-8B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAYO_QWEN_8B_I1_Q4_K_M_GGUF

    GGUF_FILE = "Sayo-Qwen-8B.i1-Q4_K_M.gguf"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher Sayo-Qwen-8B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Reinstall patches at load time to override any broken patch chains from
        # alphabetically-later loaders imported before this one.
        _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        # GGUF contains only LLM weights; vision encoder weights are absent/mismatched
        model_kwargs.setdefault("ignore_mismatched_sizes", True)
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # This GGUF contains only language-model weights (no vision encoder).
        # Use a text-only message so the forward pass skips the vision path.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe a beautiful sunset over the ocean.",
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
