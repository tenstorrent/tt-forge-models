# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 8B Thinking Heretic i1 GGUF model loader implementation for
image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
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

    @staticmethod
    def _apply_gguf_compat_patches():
        """Patch is_gguf_available and load_gguf_checkpoint for transformers 5.2+.

        transformers 5.2 calls version.parse(gguf_version) which raises
        InvalidVersion when gguf is installed at runtime and its metadata is not
        yet visible to importlib.metadata (stale cache). Wrap the function to
        fall back to a find_spec check on failure."""
        import inspect
        import importlib.util

        import transformers.modeling_gguf_pytorch_utils as _gguf_mod
        import transformers.utils.import_utils as _import_utils

        _orig_is_gguf_available = _import_utils.is_gguf_available

        def _safe_is_gguf_available(min_version=None):
            try:
                if min_version is not None:
                    return _orig_is_gguf_available(min_version=min_version)
                return _orig_is_gguf_available()
            except Exception:
                return importlib.util.find_spec("gguf") is not None

        _import_utils.is_gguf_available = _safe_is_gguf_available
        _gguf_mod.is_gguf_available = _safe_is_gguf_available

        chain_top = _gguf_mod.load_gguf_checkpoint
        try:
            needs_compat = (
                "model_to_load" not in inspect.signature(chain_top).parameters
            )
        except Exception:
            needs_compat = False

        if needs_compat:

            def _compat_load_gguf(gguf_path, return_tensors=False, **kw):
                kw.pop("model_to_load", None)
                return chain_top(gguf_path, return_tensors=return_tensors, **kw)

            orig_get_map = _gguf_mod.get_gguf_hf_weights_map

            def _compat_get_map(
                hf_model, processor, model_type=None, num_layers=None, qual_name=""
            ):
                if hf_model is None:
                    return {}
                return orig_get_map(
                    hf_model, processor, model_type, num_layers, qual_name
                )

            _gguf_mod.load_gguf_checkpoint = _compat_load_gguf
            _gguf_mod.get_gguf_hf_weights_map = _compat_get_map

    def load_model(self, *, dtype_override=None, **kwargs):
        self._apply_gguf_compat_patches()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")

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
