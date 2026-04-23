# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/AdaVaR-7B-i1-GGUF model loader implementation for image to text.
"""

import os

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
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
    """Available mradermacher AdaVaR-7B-i1-GGUF variants for image to text."""

    ADAVAR_7B_I1_Q4_K_M_GGUF = "7B_i1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher AdaVaR-7B-i1-GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ADAVAR_7B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/AdaVaR-7B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ADAVAR_7B_I1_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.ADAVAR_7B_I1_Q4_K_M_GGUF: "AdaVaR-7B.i1-Q4_K_M.gguf",
    }

    BASE_MODEL = "ZejunLi/AdaVaR-7B"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    @property
    def _gguf_file(self):
        return self._GGUF_FILES[self._variant]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher AdaVaR-7B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # GGUF repo ships only weights; load processor from the base model.
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.BASE_MODEL,
                cache_dir=self._cache_dir,
            )

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = AutoConfig.from_pretrained(
                self.BASE_MODEL, cache_dir=self._cache_dir
            )
            if self.num_layers is not None:
                if hasattr(config, "text_config"):
                    config.text_config.num_hidden_layers = self.num_layers
                else:
                    config.num_hidden_layers = self.num_layers
            model = AutoModelForImageTextToText.from_config(config)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model_kwargs["gguf_file"] = self._gguf_file

            if self.num_layers is not None:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self._gguf_file
                )
                config.num_hidden_layers = self.num_layers
                model_kwargs["config"] = config

            model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.BASE_MODEL,
                cache_dir=self._cache_dir,
            )

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

    @property
    def _cache_dir(self):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            return "/tmp/tt_forge_model_cache"
        return None

    def load_config(self):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            self.config = AutoConfig.from_pretrained(
                self.BASE_MODEL, cache_dir=self._cache_dir
            )
        else:
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
            )
        return self.config
