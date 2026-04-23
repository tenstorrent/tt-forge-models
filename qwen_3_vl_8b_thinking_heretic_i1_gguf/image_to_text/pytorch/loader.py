# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 8B Thinking Heretic i1 GGUF model loader implementation for
image to text.
"""
import importlib.metadata

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
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

# Stash the model_to_load across the old-style patch chain that strips it.
_pending_model_to_load = None


def _refresh_gguf_detection():
    """Refresh transformers' gguf package detection if the package was installed after import."""
    from transformers.utils import import_utils

    if "gguf" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        import_utils.is_gguf_available.cache_clear()


def _patch_qwen3vl_support():
    """Register qwen3vl GGUF architecture as alias for qwen3 config mapping.

    transformers 5.x does not yet recognise the qwen3vl GGUF architecture used
    by Qwen3-VL GGUF checkpoints.  This patches the supported-architectures list
    and the config-field mapping so that load_gguf_checkpoint accepts the file.
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen3vl",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support.

    Some older patches in the chain have the signature
    ``(gguf_path, return_tensors=False)`` and will raise TypeError if
    ``model_to_load`` is forwarded.  We stash it in a module-level variable so
    that ``_patched_get_gguf_hf_weights_map`` can recover it even after the
    inner chain drops it.
    """
    global _pending_model_to_load
    _patch_qwen3vl_support()
    # Strip model_to_load before entering the inner chain – old-style patches
    # do not accept the kwarg and will raise TypeError.
    _pending_model_to_load = kwargs.pop("model_to_load", None)
    try:
        result = _orig_load_gguf_checkpoint(*args, **kwargs)
    finally:
        _pending_model_to_load = None
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3_vl"
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to map qwen3_vl HF model type to qwen3vl gguf arch.

    When old-style patches strip model_to_load from the call chain, the real
    load_gguf_checkpoint receives model_to_load=None.  We recover the stashed
    value from _pending_model_to_load so tensor key mapping still works.
    """
    if hf_model is None and _pending_model_to_load is not None:
        hf_model = _pending_model_to_load
    if model_type is None and hf_model is not None:
        model_type = hf_model.config.model_type
    if model_type in ("qwen3_vl", "qwen3_vl_text"):
        model_type = "qwen3vl"
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_patch_qwen3vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Qwen 3 VL 8B Thinking Heretic i1 GGUF variants for image to text."""

    QWEN_3_VL_8B_THINKING_HERETIC_I1_Q4_K_M_GGUF = "8b_thinking_heretic_i1_q4_k_m_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 8B Thinking Heretic i1 GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_8B_THINKING_HERETIC_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-8B-Thinking-heretic-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_8B_THINKING_HERETIC_I1_Q4_K_M_GGUF

    GGUF_FILE = "Qwen3-VL-8B-Thinking-heretic.i1-Q4_K_M.gguf"

    # GGUF repos do not ship a config; use the base model for config and processor.
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Thinking"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 8B Thinking Heretic i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)

        # Pre-load full VL config: GGUF metadata lacks vision_config needed by
        # Qwen3VLForConditionalGeneration, so pass the base model config explicitly
        # to bypass GGUF config extraction.
        model_kwargs["config"] = AutoConfig.from_pretrained(self.BASE_MODEL)

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
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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
