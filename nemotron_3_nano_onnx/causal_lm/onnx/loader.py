# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron 3 Nano ONNX model loader implementation for causal language modeling.
"""

from typing import Optional

import onnx
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Nemotron 3 Nano ONNX model variants for causal language modeling."""

    NEMOTRON_3_NANO_4B_BF16_ONNX = "4B_BF16_ONNX"


class ModelLoader(ForgeModel):
    """Nemotron 3 Nano ONNX model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_NANO_4B_BF16_ONNX: ModelConfig(
            pretrained_model_name="onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_NANO_4B_BF16_ONNX

    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Nemotron 3 Nano ONNX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the Nemotron 3 Nano ONNX model.

        Returns:
            onnx.ModelProto: The ONNX model instance.
        """
        local_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="onnx/model.onnx",
        )
        model = onnx.load(local_path)

        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the Nemotron 3 Nano ONNX model.

        Returns:
            dict: Tokenized input tensors with input_ids and attention_mask.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
            )

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        return inputs
