# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen2.5-0.5B-Instruct LiteRT model loader implementation for causal language modeling.

Distributed as TFLite files via litert-community/Qwen2.5-0.5B-Instruct on HuggingFace.
The loader downloads the TFLite model and wraps it in a PyTorch-compatible interface,
sourcing the tokenizer from the Qwen/Qwen2.5-0.5B-Instruct base model.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
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
    """Available Qwen2.5-0.5B-Instruct LiteRT model variants for causal language modeling."""

    QWEN2_5_0_5B_INSTRUCT_LITERT_SEQ128_Q8 = "Qwen2.5-0.5B-Instruct-seq128-q8"


class QwenLiteRTWrapper(nn.Module):
    """PyTorch wrapper around a Qwen2.5 LiteRT TFLite model for causal language modeling.

    Uses the LiteRT prefill+decode pipeline with external KV cache management.
    The prefill signature processes the full input sequence, then decode produces
    one step of logits.
    """

    PREFILL_SIG = "prefill_128"
    PREFILL_LEN = 128

    def __init__(self, tflite_model_path: str):
        super().__init__()
        from ai_edge_litert import interpreter as interpreter_lib
        from ai_edge_litert import pywrap_genai_ops

        self.interpreter = interpreter_lib.InterpreterWithCustomOps(
            custom_op_registerers=[pywrap_genai_ops.GenAIOpsRegisterer],
            model_path=tflite_model_path,
            num_threads=2,
            experimental_default_delegate_latest_features=True,
        )
        self._prefill_runner = self.interpreter.get_signature_runner(self.PREFILL_SIG)
        self._decode_runner = self.interpreter.get_signature_runner("decode")

        prefill_inputs = self._prefill_runner.get_input_details()
        self._num_layers = sum(1 for k in prefill_inputs if k.startswith("kv_cache_k_"))
        self._kv_shape = tuple(prefill_inputs["kv_cache_k_0"]["shape"].tolist())

    def _init_kv_cache(self) -> dict:
        return {
            f"kv_cache_{t}_{i}": np.zeros(self._kv_shape, dtype=np.float32)
            for i in range(self._num_layers)
            for t in ("k", "v")
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        ids_np = input_ids[0].numpy().astype(np.int32)

        tokens = np.zeros([1, self.PREFILL_LEN], dtype=np.int32)
        fill = min(seq_len, self.PREFILL_LEN)
        tokens[0, :fill] = ids_np[:fill]

        input_pos = np.zeros([self.PREFILL_LEN], dtype=np.int32)
        input_pos[:fill] = np.arange(fill, dtype=np.int32)

        kv_cache = self._init_kv_cache()

        prefill_out = self._prefill_runner(
            tokens=tokens, input_pos=input_pos, **kv_cache
        )
        kv_cache = dict(prefill_out)

        last_token = ids_np[fill - 1 : fill].reshape(1, 1)
        decode_pos = np.array([fill - 1], dtype=np.int32)

        decode_out = self._decode_runner(
            tokens=last_token, input_pos=decode_pos, **kv_cache
        )
        return torch.from_numpy(decode_out["logits"].copy())


class ModelLoader(ForgeModel):
    """Qwen2.5-0.5B-Instruct LiteRT model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN2_5_0_5B_INSTRUCT_LITERT_SEQ128_Q8: ModelConfig(
            pretrained_model_name="litert-community/Qwen2.5-0.5B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN2_5_0_5B_INSTRUCT_LITERT_SEQ128_Q8

    TFLITE_FILENAME = "Qwen2.5-0.5B-Instruct_seq128_q8_ekv1280.tflite"

    TOKENIZER_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

    SEQ_LEN = 128

    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

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

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_REPO)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        repo_id = self._variant_config.pretrained_model_name

        tflite_path = hf_hub_download(repo_id=repo_id, filename=self.TFLITE_FILENAME)

        model = QwenLiteRTWrapper(tflite_path)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.SEQ_LEN,
        )

        input_ids = encoded["input_ids"].to(torch.int32)
        input_ids = input_ids.repeat_interleave(batch_size, dim=0)

        return {"input_ids": input_ids}
