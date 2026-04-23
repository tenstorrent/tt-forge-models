# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF model loader implementation for image to text.
"""

import os

from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
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
    """Available mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF model variants for image to text."""

    SEA_LION_VL_IT_MERGE_100226_I1_GGUF = "SEA_LION_VL_IT_Merge_100226_i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF model loader implementation for image to text tasks."""

    # Base model provides the config; GGUF repo only contains quantized weights
    _VARIANTS = {
        ModelVariant.SEA_LION_VL_IT_MERGE_100226_I1_GGUF: LLMModelConfig(
            pretrained_model_name="SEACrowd/SEA-LION-VL-IT-Merge-100226",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEA_LION_VL_IT_MERGE_100226_I1_GGUF

    GGUF_REPO = "mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF"
    GGUF_FILE = "SEA-LION-VL-IT-Merge-100226.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher SEA-LION-VL-IT-Merge-100226 i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = AutoConfig.from_pretrained(pretrained_model_name)
            model = AutoModelForImageTextToText.from_config(config, **model_kwargs)
        else:
            gguf_path = hf_hub_download(
                repo_id=self.GGUF_REPO,
                filename=self.GGUF_FILE,
            )
            model_kwargs["gguf_file"] = gguf_path
            model_kwargs |= kwargs
            model = AutoModelForImageTextToText.from_pretrained(
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
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
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
