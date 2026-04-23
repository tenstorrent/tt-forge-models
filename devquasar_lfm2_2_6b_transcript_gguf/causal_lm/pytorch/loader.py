# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DevQuasar LFM2 2.6B Transcript GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter
from typing import Optional

# LFM2 uses a GPT2-BPE tokenizer with ChatML special tokens — same shape as Qwen2.
# transformers 5.x has the lfm2 model/config registered but is missing the tokenizer
# converter entry; register it here before any tokenizer load.
if "lfm2" not in GGUF_TO_FAST_CONVERTERS:
    GGUF_TO_FAST_CONVERTERS["lfm2"] = GGUFQwen2Converter

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
    """Available DevQuasar LFM2 2.6B Transcript GGUF model variants for causal language modeling."""

    LFM2_2_6B_TRANSCRIPT_Q4_K_M_GGUF = "2_6B_Transcript_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """DevQuasar LFM2 2.6B Transcript GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LFM2_2_6B_TRANSCRIPT_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="DevQuasar/LiquidAI.LFM2-2.6B-Transcript-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_2_6B_TRANSCRIPT_Q4_K_M_GGUF

    GGUF_FILE = "LiquidAI.LFM2-2.6B-Transcript.Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DevQuasar LFM2 2.6B Transcript GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        # The GGUF tokenizer_config only provides integer IDs (bos=1, eos=7, pad=0),
        # not string tokens, so the loader falls back to <s>/</ s> defaults which
        # are NOT in the LFM2 vocab — they get appended as out-of-range IDs.
        # Fix: map the integer IDs from GGUF metadata to the actual token strings.
        self.tokenizer.bos_token = self.tokenizer.convert_ids_to_tokens(
            1
        )  # <|startoftext|>
        self.tokenizer.eos_token = self.tokenizer.convert_ids_to_tokens(7)  # <|im_end|>
        self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)  # <|pad|>

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
