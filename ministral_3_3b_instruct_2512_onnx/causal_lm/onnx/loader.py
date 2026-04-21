# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral-3-3B-Instruct-2512 ONNX model loader implementation for causal language modeling.
"""

from typing import Optional

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoProcessor

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
    """Available Ministral-3-3B-Instruct-2512 ONNX model variants for causal language modeling."""

    MINISTRAL_3_3B_INSTRUCT_2512_ONNX = "Ministral_3_3B_Instruct_2512_ONNX"


class ModelLoader(ForgeModel):
    """Ministral-3-3B-Instruct-2512 ONNX model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_3B_INSTRUCT_2512_ONNX: ModelConfig(
            pretrained_model_name="mistralai/Ministral-3-3B-Instruct-2512-ONNX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_3B_INSTRUCT_2512_ONNX

    # Quantized decoder variant referenced in the upstream ONNXRuntime example.
    _DECODER_ONNX_FILENAME = "onnx/decoder_model_merged_q4.onnx"
    _DECODER_DATA_FILES = (
        "onnx/decoder_model_merged_q4.onnx_data",
        "onnx/decoder_model_merged_q4.onnx_data_1",
    )

    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Ministral-3-3B-Instruct-2512 ONNX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_config(self):
        if self.config is None:
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self.config

    def load_model(self, **kwargs):
        """Download and load the Ministral-3-3B-Instruct-2512 ONNX decoder model.

        The decoder uses external ONNX data files which must be resolved in the
        same cache directory as the graph file before loading.

        Returns:
            onnx.ModelProto: The loaded ONNX decoder model.
        """
        repo_id = self._variant_config.pretrained_model_name

        for data_filename in self._DECODER_DATA_FILES:
            hf_hub_download(repo_id=repo_id, filename=data_filename)

        onnx_path = hf_hub_download(
            repo_id=repo_id, filename=self._DECODER_ONNX_FILENAME
        )
        model = onnx.load(onnx_path)
        return model

    def load_inputs(self, **kwargs):
        """Build sample decoder inputs matching the ONNX graph signature.

        The decoder expects pre-embedded token inputs plus an empty past_key_values
        cache sized from the text config.

        Returns:
            dict: Inputs for the ONNX decoder session.
        """
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        config = self._load_config()
        text_config = config.text_config

        messages = [{"role": "user", "content": self.sample_text}]
        tokenized = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
        )

        input_ids = tokenized["input_ids"].astype(np.int64)
        attention_mask = tokenized["attention_mask"].astype(np.int64)
        batch_size, seq_len = input_ids.shape

        hidden_size = text_config.hidden_size
        inputs_embeds = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float32)

        position_ids = np.tile(np.arange(seq_len, dtype=np.int64), (batch_size, 1))

        past_key_values = {
            f"past_key_values.{layer}.{kv}": np.zeros(
                (
                    batch_size,
                    text_config.num_key_value_heads,
                    0,
                    text_config.head_dim,
                ),
                dtype=np.float32,
            )
            for layer in range(text_config.num_hidden_layers)
            for kv in ("key", "value")
        }

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **past_key_values,
        }
