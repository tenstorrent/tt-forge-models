# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ASID Captioner 3B i1 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen2VLForConditionalGeneration,
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
    """Available ASID Captioner 3B i1 GGUF model variants for image to text."""

    ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF = "3b_i1_Q4_K_M_gguf"


class ModelLoader(ForgeModel):
    """ASID Captioner 3B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/ASID-Captioner-3B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF

    GGUF_FILE = "ASID-Captioner-3B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ASID Captioner 3B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_gguf_qwen2vl_support():
        """Patch transformers to support qwen2vl GGUF loading (not yet upstream)."""
        try:
            import importlib.metadata

            import gguf as _gguf

            # gguf lacks __version__; set it so transformers' is_gguf_available() can parse it
            if not hasattr(_gguf, "__version__"):
                _gguf.__version__ = importlib.metadata.version("gguf")

            from transformers.integrations import ggml as _ggml
            from transformers.integrations.ggml import GGUFQwen2Converter
            import transformers.modeling_gguf_pytorch_utils as _gguf_utils

            if "qwen2vl" not in _ggml.GGUF_CONFIG_MAPPING:
                _ggml.GGUF_CONFIG_MAPPING["qwen2vl"] = dict(
                    _ggml.GGUF_CONFIG_MAPPING["qwen2"]
                )
            if "qwen2vl" not in _ggml.GGUF_TO_FAST_CONVERTERS:
                _ggml.GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter
            if "qwen2vl" not in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING.get(
                "config", {}
            ):
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
                    "qwen2vl"
                ] = _ggml.GGUF_CONFIG_MAPPING["qwen2vl"]
            if "qwen2vl" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
                _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

            # Patch get_gguf_hf_weights_map to map qwen2_vl -> qwen2vl and
            # resolve num_hidden_layers from text_config sub-config
            _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

            def _patched_get_map(
                hf_model, processor, model_type=None, num_layers=None, qual_name=""
            ):
                cfg = getattr(hf_model, "config", None)
                if getattr(cfg, "model_type", None) == "qwen2_vl":
                    if model_type is None:
                        model_type = "qwen2vl"
                    # Qwen2VLConfig nests num_hidden_layers inside text_config
                    if num_layers is None:
                        num_layers = getattr(
                            getattr(cfg, "text_config", None), "num_hidden_layers", None
                        )
                elif model_type == "qwen2_vl":
                    model_type = "qwen2vl"
                return _orig_get_map(
                    hf_model, processor, model_type, num_layers, qual_name
                )

            _gguf_utils.get_gguf_hf_weights_map = _patched_get_map
        except Exception:
            pass

    def load_model(self, *, dtype_override=None, **kwargs):
        self._patch_gguf_qwen2vl_support()

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(
            "AudioVisual-Caption/ASID-Captioner-3B"
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
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
