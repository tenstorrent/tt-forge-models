# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen2.5-0.5B-Instruct LiteRT model loader implementation for causal language modeling.

Distributed as TFLite files via litert-community/Qwen2.5-0.5B-Instruct on HuggingFace.
The loader downloads the TFLite decode-step model and wraps it in a PyTorch-compatible
interface. The model accepts a single token and position index per call, with external
KV cache (ekv1280) tensors managed internally.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

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
    """Available Qwen2.5-0.5B-Instruct LiteRT model variants for causal language modeling."""

    QWEN2_5_0_5B_INSTRUCT_LITERT_SEQ128_Q8 = "Qwen2.5-0.5B-Instruct-seq128-q8"


class QwenLiteRTWrapper(nn.Module):
    """PyTorch wrapper around a Qwen2.5 LiteRT TFLite decode-step model.

    The TFLite model is a single-token decode model with external KV cache.
    Inputs: decode_tokens [1,1], decode_input_pos [1], plus 48 KV cache tensors.
    Output: logits [1, 1, vocab_size].
    """

    def __init__(self, tflite_model_path: str):
        super().__init__()
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self._token_idx = next(
            d["index"] for d in input_details if "decode_tokens" in d["name"]
        )
        self._pos_idx = next(
            d["index"] for d in input_details if "decode_input_pos" in d["name"]
        )
        self._kv_inputs = [
            (d["index"], tuple(d["shape"]))
            for d in input_details
            if "kv_cache" in d["name"]
        ]
        # Logits output is the 3-D tensor [1, 1, vocab_size]
        self._logits_idx = next(
            d["index"] for d in output_details if len(d["shape"]) == 3
        )

    def forward(
        self, decode_tokens: torch.Tensor, decode_input_pos: torch.Tensor
    ) -> torch.Tensor:
        for idx, shape in self._kv_inputs:
            self.interpreter.set_tensor(idx, np.zeros(shape, dtype=np.float32))

        self.interpreter.set_tensor(
            self._token_idx, decode_tokens.numpy().astype(np.int32)
        )
        self.interpreter.set_tensor(
            self._pos_idx, decode_input_pos.numpy().astype(np.int32)
        )
        self.interpreter.invoke()

        return torch.from_numpy(self.interpreter.get_tensor(self._logits_idx).copy())


class ModelLoader(ForgeModel):
    """Qwen2.5-0.5B-Instruct LiteRT model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN2_5_0_5B_INSTRUCT_LITERT_SEQ128_Q8: ModelConfig(
            pretrained_model_name="litert-community/Qwen2.5-0.5B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN2_5_0_5B_INSTRUCT_LITERT_SEQ128_Q8

    TFLITE_FILENAME = "Qwen2.5-0.5B-Instruct_seq128_q8_ekv1280.tflite"

    def load_model(self, *, dtype_override=None, **kwargs):
        repo_id = self._variant_config.pretrained_model_name
        tflite_path = hf_hub_download(repo_id=repo_id, filename=self.TFLITE_FILENAME)
        model = QwenLiteRTWrapper(tflite_path)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        return {
            "decode_tokens": torch.zeros(batch_size, 1, dtype=torch.int32),
            "decode_input_pos": torch.zeros(1, dtype=torch.int32),
        }

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen2.5-0.5B-Instruct LiteRT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )
