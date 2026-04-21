# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 2B ONNX model loader implementation for causal language modeling.
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
    """Available Qwen 3.5 2B ONNX model variants for causal language modeling."""

    QWEN_3_5_2B_ONNX = "2B_ONNX"


class ModelLoader(ForgeModel):
    """Qwen 3.5 2B ONNX model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_2B_ONNX: ModelConfig(
            pretrained_model_name="onnx-community/Qwen3.5-2B-ONNX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_2B_ONNX

    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3.5 2B ONNX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    # The decoder_model_merged.onnx proto references external weight shards that
    # onnx.load expects to find alongside the graph file.
    _EXTERNAL_DATA_FILES = (
        "onnx/decoder_model_merged.onnx_data",
        "onnx/decoder_model_merged.onnx_data_1",
        "onnx/decoder_model_merged.onnx_data_2",
        "onnx/decoder_model_merged.onnx_data_3",
    )

    def load_model(self, **kwargs):
        """Load and return the Qwen 3.5 2B ONNX model.

        Returns:
            onnx.ModelProto: The ONNX model instance.
        """
        repo_id = self._variant_config.pretrained_model_name
        for data_file in self._EXTERNAL_DATA_FILES:
            hf_hub_download(repo_id=repo_id, filename=data_file)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename="onnx/decoder_model_merged.onnx",
        )
        model = onnx.load(local_path)

        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the Qwen 3.5 2B ONNX model.

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
            enable_thinking=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        return inputs
