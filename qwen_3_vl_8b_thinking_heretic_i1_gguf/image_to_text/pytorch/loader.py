# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 8B Thinking Heretic i1 GGUF model loader implementation for
image to text.
"""
import os

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig,
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
    def _use_random_weights():
        return os.environ.get("TT_RANDOM_WEIGHTS") or os.environ.get(
            "TT_COMPILE_ONLY_SYSTEM_DESC"
        )

    def _qwen3vl_config(self):
        from transformers import Qwen3VLTextConfig

        text_config = Qwen3VLTextConfig(
            hidden_size=4096,
            intermediate_size=22016,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
        )
        return Qwen3VLConfig(text_config=text_config)

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._use_random_weights():
            config = self._qwen3vl_config()
            target_dtype = (
                dtype_override if dtype_override is not None else torch.bfloat16
            )
            orig_dtype = torch.get_default_dtype()
            torch.set_default_dtype(target_dtype)
            try:
                model = Qwen3VLForConditionalGeneration(config)
            finally:
                torch.set_default_dtype(orig_dtype)
            model.eval()
            return model

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
        max_length = self._variant_config.max_length

        if self._use_random_weights():
            vocab_size = 151936
            input_ids = torch.randint(0, vocab_size, (batch_size, max_length))
            attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")

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
