# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF model loader implementation for image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
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


def _patch_qwen2vl_support():
    """Register qwen2vl architecture as an alias for qwen2.

    Qwen2.5-VL GGUF files declare architecture as 'qwen2vl', which
    transformers 5.x does not yet recognise for GGUF loading.
    """
    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen2" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen2vl",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen2"],
            )
    if "qwen2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen2vl", GGUF_TO_FAST_CONVERTERS["qwen2"])


def _find_true_load_gguf_checkpoint():
    """Walk the patch chain to find the real load_gguf_checkpoint from transformers.

    Loaders patch _gguf_utils.load_gguf_checkpoint using different variable names
    and sometimes closures.  We identify the real function by its source file.
    """
    import os

    real_file = _gguf_utils.__file__

    _CANDIDATE_NAMES = [
        "_orig_load_gguf_checkpoint",
        "orig_load",
        "_orig",
        "orig",
        "_original_load_gguf_checkpoint",
    ]

    fn = _gguf_utils.load_gguf_checkpoint
    seen = set()

    while id(fn) not in seen:
        seen.add(id(fn))

        # If this function lives in the real transformers module, we're done.
        code = getattr(fn, "__code__", None)
        if code and os.path.samefile(code.co_filename, real_file):
            return fn

        # Try well-known global names that loaders use to stash the previous fn.
        found = False
        for name in _CANDIDATE_NAMES:
            _next = fn.__globals__.get(name)
            if _next is not None and callable(_next) and id(_next) not in seen:
                fn = _next
                found = True
                break

        # Also follow closure cells — our own wrapper captures true_orig that way.
        if not found and fn.__closure__:
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if callable(val) and getattr(val, "__code__", None) and id(val) not in seen:
                    fn = val
                    found = True
                    break

        if not found:
            break

    return fn


def _make_patched(true_orig):
    def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
        _patch_qwen2vl_support()
        result = true_orig(gguf_path, return_tensors=return_tensors, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen2vl":
            result["config"]["model_type"] = "qwen2_5_vl"
        return result

    return _patched_load_gguf_checkpoint


def _apply_patches():
    """Install all monkey-patches into the transformers GGUF utils module."""
    _patch_qwen2vl_support()

    # Patch load_gguf_checkpoint to bypass broken intermediate wrappers.
    fresh = _make_patched(_find_true_load_gguf_checkpoint())
    _gguf_utils.load_gguf_checkpoint = fresh
    _config_utils.load_gguf_checkpoint = fresh
    _auto_tokenizer.load_gguf_checkpoint = fresh
    _tok_utils.load_gguf_checkpoint = fresh

    # Patch get_gguf_hf_weights_map to handle Qwen2.5-VL configs.
    # Qwen2_5_VLConfig stores num_hidden_layers in text_config, not at the top
    # level, and the GGUF arch name is 'qwen2vl', not 'qwen2_5_vl'.
    _orig_weights_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        cfg = getattr(hf_model, "config", None)
        if cfg is not None and getattr(cfg, "model_type", None) == "qwen2_5_vl":
            if model_type is None:
                model_type = "qwen2vl"
            if num_layers is None:
                text_cfg = getattr(cfg, "text_config", None)
                if text_cfg is not None:
                    num_layers = getattr(text_cfg, "num_hidden_layers", None)
        return _orig_weights_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_apply_patches()


class ModelVariant(StrEnum):
    """Available Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF variants for image to text."""

    JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF = "7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/JSL-VL-7B-MedAgentBench-v2-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF: "JSL-VL-7B-MedAgentBench-v2.i1-Q4_K_M.gguf",
    }

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
        )

        # Re-apply all patches before loading so they're active when
        # from_pretrained calls load_gguf_checkpoint / get_gguf_hf_weights_map.
        _apply_patches()

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
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
