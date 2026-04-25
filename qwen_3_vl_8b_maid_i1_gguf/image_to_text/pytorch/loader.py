# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 8B Maid i1 GGUF model loader implementation for image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
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


def _patch_qwen3vl_gguf_out_hidden_size():
    """Fix vision out_hidden_size for qwen3_vl GGUF models.

    The gguf-py default for Qwen3VLVisionConfig.out_hidden_size is 3584, but
    for Qwen3-VL-8B it must equal the text hidden_size (4096). Patch
    load_gguf_checkpoint to correct this before the model is built.
    """
    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3_vl":
            vision_cfg = config.get("vision_config", {})
            text_cfg = config.get("text_config", {})
            lm_hidden = text_cfg.get("hidden_size") or config.get("hidden_size")
            if lm_hidden and vision_cfg.get("out_hidden_size", 3584) != lm_hidden:
                vision_cfg = dict(vision_cfg)
                vision_cfg["out_hidden_size"] = lm_hidden
                config = dict(config)
                config["vision_config"] = vision_cfg
                result = dict(result)
                result["config"] = config
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_qwen3vl_gguf_out_hidden_size()


class ModelVariant(StrEnum):
    """Available Qwen 3 VL 8B Maid i1 GGUF model variants for image to text."""

    QWEN_3_VL_8B_MAID_I1_GGUF = "8b_maid_i1_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 8B Maid i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_8B_MAID_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-8B-maid-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_8B_MAID_I1_GGUF

    GGUF_FILE = "Qwen3-VL-8B-maid.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 8B Maid i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
