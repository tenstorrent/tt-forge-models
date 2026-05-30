# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chocolatine causal language modeling loader

Chocolatine-3B is a Phi-3 (Phi3ForCausalLM) based French/English instruct model.
This loader consumes the Q4_K_M GGUF quantization published by the author; the
weights are dequantized to a standard PyTorch Phi-3 model at load time via the
``gguf_file`` argument supported by transformers.
"""

from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    INSTRUCT_3B_DPO_Q4_K_M = "3b_instruct_dpo_revised_q4_k_m"


# GGUF file shipped in each variant's HuggingFace repo. transformers reads the
# quantized tensors from this file and dequantizes them on load.
_GGUF_FILES = {
    ModelVariant.INSTRUCT_3B_DPO_Q4_K_M: "chocolatine-3b-instruct-dpo-revised-q4_k_m.gguf",
}


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.INSTRUCT_3B_DPO_Q4_K_M: LLMModelConfig(
            pretrained_model_name="jpacifico/Chocolatine-3B-Instruct-DPO-Revised-Q4_K_M-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INSTRUCT_3B_DPO_Q4_K_M

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Chocolatine-3B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self):
        return _GGUF_FILES[self._variant]

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                gguf_file=self._gguf_file(),
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_tokenizer()

        model_kwargs = {"use_cache": False, "gguf_file": self._gguf_file()}
        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
                gguf_file=self._gguf_file(),
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, prompt: Optional[str] = None):
        self._ensure_tokenizer()
        input_prompt = [
            {
                "role": "user",
                "content": prompt
                or "Can you provide ways to eat combinations of bananas and dragonfruits?",
            }
        ]
        text = self.tokenizer.apply_chat_template(
            input_prompt, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)

        return [input_ids, attn_mask]
