# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher Huihui Qwen3-VL 8B Instruct Abliterated GGUF model loader
implementation for image to text.
"""

import inspect
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from typing import Optional

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_qwen3vl_support():
    """Register qwen3vl as an alias for qwen3 in the GGUF loader.

    Qwen3-VL GGUF files declare architecture as 'qwen3vl', which transformers
    does not yet recognise. The LM config keys are identical to qwen3, so we
    alias them here and fix model_type to 'qwen3_vl' post-load.
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen3vl",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUF_TO_FAST_CONVERTERS["qwen3"])


def _find_orig_load_fn():
    """Traverse the chain of patched load_gguf_checkpoint functions to find
    the actual transformers original (identified by its __module__).

    Several other model loaders replace _gguf_utils.load_gguf_checkpoint with
    restricted-signature wrappers. We follow globals (named
    _orig_load_gguf_checkpoint) and closure variables until we reach the
    function defined in 'transformers.modeling_gguf_pytorch_utils'.
    """
    fn = _gguf_utils.load_gguf_checkpoint
    seen = set()
    while True:
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)
        if (
            getattr(fn, "__module__", None)
            == "transformers.modeling_gguf_pytorch_utils"
        ):
            return fn
        # Follow globals reference (most common pattern)
        nxt = fn.__globals__.get("_orig_load_gguf_checkpoint")
        if nxt is not None and callable(nxt) and id(nxt) != fn_id:
            fn = nxt
            continue
        # Follow closure variables (e.g. `orig_load = gguf_utils.load_gguf_checkpoint`)
        if fn.__closure__:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if callable(val) and id(val) != fn_id and id(val) not in seen:
                    fn = val
                    break
            else:
                break
            continue
        break
    return fn


def _make_patched_fn(underlying_fn):
    def _patched_load_gguf_checkpoint(*args, **kwargs):
        _patch_qwen3vl_support()
        result = underlying_fn(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen3vl":
            result["config"]["model_type"] = "qwen3_vl"
        return result

    return _patched_load_gguf_checkpoint


_patch_qwen3vl_support()


class ModelVariant(StrEnum):
    """Available Mradermacher Huihui Qwen3-VL 8B Abliterated GGUF variants for image to text."""

    HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF = (
        "8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Mradermacher Huihui Qwen3-VL 8B Abliterated GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: "Huihui-Qwen3-VL-8B-Instruct-abliterated.Q4_K_M.gguf",
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
            model="Mradermacher Huihui Qwen3-VL 8B Abliterated GGUF",
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
            "Qwen/Qwen3-VL-8B-Instruct",
        )

        # Locate the original load_gguf_checkpoint (one that accepts model_to_load)
        # by traversing any chain of restricted-signature wrappers set by other loaders,
        # then install our patched version just before from_pretrained executes.
        orig_fn = _find_orig_load_fn()
        patched_fn = _make_patched_fn(orig_fn)
        _gguf_utils.load_gguf_checkpoint = patched_fn
        _config_utils.load_gguf_checkpoint = patched_fn
        _auto_tokenizer.load_gguf_checkpoint = patched_fn
        _tok_utils.load_gguf_checkpoint = patched_fn

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
