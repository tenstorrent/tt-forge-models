# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 32B Thinking heretic i1 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional


def _patch_transformers_qwen3vl_gguf():
    """Monkey-patch transformers to add qwen3vl GGUF architecture support.

    The qwen3vl GGUF architecture is not natively supported by transformers'
    GGUF loader. We map qwen3vl -> qwen3_vl for loading.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = dict(
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3"]
    )

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            config["model_type"] = "qwen3_vl"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.configuration_utils as config_utils
    import transformers.tokenization_utils_tokenizers as tok_utils
    import transformers.models.auto.tokenization_auto as tok_auto

    config_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    tok_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    tok_auto.load_gguf_checkpoint = patched_load_gguf_checkpoint


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
    """Available Qwen 3 VL 32B Thinking heretic i1 GGUF model variants for image to text."""

    QWEN_3_VL_32B_THINKING_HERETIC_I1_GGUF = "32b_thinking_heretic_i1_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 32B Thinking heretic i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_32B_THINKING_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-32B-Thinking-heretic-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_32B_THINKING_HERETIC_I1_GGUF

    GGUF_FILE = "Qwen3-VL-32B-Thinking-heretic.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 32B Thinking heretic i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _patch_transformers_qwen3vl_gguf()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Thinking")

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
