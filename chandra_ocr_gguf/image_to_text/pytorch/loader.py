# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chandra OCR GGUF model loader implementation for image to text.
"""

import importlib.metadata
from typing import Optional

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
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


class ModelVariant(StrEnum):
    """Available Chandra OCR GGUF model variants for image to text."""

    CHANDRA_OCR_Q4_K_M_GGUF = "Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Chandra OCR GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.CHANDRA_OCR_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="noctrex/Chandra-OCR-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHANDRA_OCR_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.CHANDRA_OCR_Q4_K_M_GGUF: "Chandra-OCR-Q4_K_M.gguf",
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
            model="Chandra OCR GGUF",
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
    def _fix_gguf_model_to_load():
        """Fix model_to_load kwarg compatibility with transformers 5.x.

        Other GGUF loaders in this repo monkey-patch load_gguf_checkpoint without
        the model_to_load parameter added in transformers 5.x, causing TypeError
        when from_pretrained passes it.

        Also wraps get_gguf_hf_weights_map to handle hf_model=None gracefully
        (returns empty mapping, acceptable in compile-only mode where weights
        are not used for accuracy comparison).
        """
        import inspect

        import transformers.modeling_gguf_pytorch_utils as _gguf_utils

        current_fn = _gguf_utils.load_gguf_checkpoint
        if "model_to_load" not in inspect.signature(current_fn).parameters:
            orig_fn = current_fn

            def _wrapped(
                gguf_checkpoint_path, return_tensors=False, model_to_load=None
            ):
                return orig_fn(gguf_checkpoint_path, return_tensors=return_tensors)

            _gguf_utils.load_gguf_checkpoint = _wrapped

        current_map_fn = _gguf_utils.get_gguf_hf_weights_map
        if not getattr(current_map_fn, "_handles_none_model", False):
            orig_map_fn = current_map_fn

            def _wrapped_map(hf_model, *args, **kwargs):
                if hf_model is None:
                    return {}
                return orig_map_fn(hf_model, *args, **kwargs)

            _wrapped_map._handles_none_model = True
            _gguf_utils.get_gguf_hf_weights_map = _wrapped_map

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()
        self._fix_gguf_model_to_load()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base Qwen3-VL model.
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
        )

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
