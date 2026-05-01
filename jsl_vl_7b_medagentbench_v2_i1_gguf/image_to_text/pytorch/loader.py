# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF model loader implementation for image to text.
"""

import contextlib
from typing import Optional

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoConfig,
    AutoProcessor,
)
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils

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


def _register_qwen2vl_gguf():
    """Register qwen2vl GGUF architecture (Qwen2-VL / Qwen2.5-VL not in transformers 5.x by default)."""
    if "qwen2vl" in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        return
    # Copy qwen2 config-key mappings for qwen2vl; VL models share the same
    # transformer text-tower hyperparameters
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "qwen2vl", _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2"]
    )
    _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    # get_gguf_hf_weights_map looks up model_type in gguf-py's MODEL_ARCH_NAMES.
    # Qwen2.5-VL has model_type "qwen2_5_vl" but gguf-py only knows "qwen2vl".
    _orig_weights_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        if hf_model is not None and hasattr(hf_model, "config"):
            cfg = hf_model.config
            # Upstream patchers (e.g. momix_44) may pre-fill model_type from
            # hf_model.config.model_type, so check the effective value, not just None.
            effective_mt = model_type if model_type is not None else getattr(cfg, "model_type", None)
            if effective_mt in ("qwen2_5_vl", "qwen2_vl"):
                model_type = "qwen2vl"
                # VLM config nests num_hidden_layers in text_config; top-level
                # raises AttributeError, so pull it from text_config directly.
                if num_layers is None:
                    text_cfg = getattr(cfg, "text_config", None)
                    if text_cfg is not None:
                        num_layers = getattr(text_cfg, "num_hidden_layers", None)
        return _orig_weights_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_weights_map


def _find_real_load_gguf_checkpoint():
    """Walk the module-patching chain to find the real transformers load_gguf_checkpoint."""
    current = _gguf_utils.load_gguf_checkpoint
    seen = set()
    while True:
        fn_id = id(current)
        if fn_id in seen:
            break
        seen.add(fn_id)
        if (
            getattr(current, "__module__", None) == "transformers.modeling_gguf_pytorch_utils"
            and getattr(current, "__qualname__", None) == "load_gguf_checkpoint"
        ):
            return current
        # Each patcher stores the previous function as _orig_load_gguf_checkpoint
        # in its module globals, or sometimes as orig_load.
        g = getattr(current, "__globals__", {})
        nxt = None
        for key in ("_orig_load_gguf_checkpoint", "orig_load"):
            v = g.get(key)
            if callable(v) and id(v) not in seen:
                nxt = v
                break
        if nxt is None:
            # Try closures as fallback
            for cell in getattr(current, "__closure__", None) or ():
                try:
                    v = cell.cell_contents
                    if callable(v) and id(v) not in seen:
                        nxt = v
                        break
                except ValueError:
                    pass
        if nxt is None:
            break
        current = nxt
    return current


@contextlib.contextmanager
def _qwen2vl_gguf_context():
    """Bypass the broken patcher chain for load_gguf_checkpoint.

    Every GGUF loader patches load_gguf_checkpoint but drops model_to_load,
    which breaks return_tensors=True.  This context manager replaces all four
    binding sites with a wrapper that calls the real transformers function
    directly, then remaps qwen2vl → qwen2_5_vl in the returned config.
    """
    real_fn = _find_real_load_gguf_checkpoint()

    def _wrapped(gguf_path, return_tensors=False, model_to_load=None):
        result = real_fn(
            gguf_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )
        if isinstance(result.get("config"), dict) and result["config"].get("model_type") == "qwen2vl":
            result["config"]["model_type"] = "qwen2_5_vl"
        return result

    orig_gguf = _gguf_utils.load_gguf_checkpoint
    orig_config = _config_utils.load_gguf_checkpoint
    orig_auto_tok = _auto_tokenizer.load_gguf_checkpoint
    orig_tok = _tok_utils.load_gguf_checkpoint

    _gguf_utils.load_gguf_checkpoint = _wrapped
    _config_utils.load_gguf_checkpoint = _wrapped
    _auto_tokenizer.load_gguf_checkpoint = _wrapped
    _tok_utils.load_gguf_checkpoint = _wrapped
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = orig_gguf
        _config_utils.load_gguf_checkpoint = orig_config
        _auto_tokenizer.load_gguf_checkpoint = orig_auto_tok
        _tok_utils.load_gguf_checkpoint = orig_tok


_register_qwen2vl_gguf()


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

        # use_fast=False: transformers 5.x defaults Qwen2VLImageProcessor to
        # fast mode which breaks slow-checkpoint loading
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            use_fast=False,
        )

        # Load config from the base model so VL-specific fields (mrope_section
        # etc.) are present; the GGUF file only carries text-transformer keys.
        base_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        with _qwen2vl_gguf_context():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name, config=base_config, **model_kwargs
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
