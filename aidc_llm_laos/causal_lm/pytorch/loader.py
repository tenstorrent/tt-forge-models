# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AIDC-LLM-Laos model loader implementation for causal language modeling.

This loader targets the GGUF distribution
``mradermacher/aidc-llm-laos-4b-GGUF`` which is a quantized export of the
Gemma 3 (gemma3_text) based model ``AIDC-LAOS/aidc-llm-laos-4b``. The GGUF
weights are de-quantized to a regular ``Gemma3ForCausalLM`` torch module by
transformers via the ``gguf_file`` argument.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

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
    """Available AIDC-LLM-Laos model variants for causal LM."""

    LAOS_4B_GGUF = "4b_gguf"


# GGUF file (within the GGUF repo) to de-quantize for each variant.
_GGUF_FILES = {
    ModelVariant.LAOS_4B_GGUF: "aidc-llm-laos-4b.Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """AIDC-LLM-Laos model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LAOS_4B_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/aidc-llm-laos-4b-GGUF",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LAOS_4B_GGUF

    sample_text = "What is the capital of Laos?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="AIDC LLM Laos",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current causal_lm variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = _GGUF_FILES[self._variant]
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AIDC-LLM-Laos causal_lm model instance.

        The GGUF weights are de-quantized into a ``Gemma3ForCausalLM`` module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                dtype. If not provided, transformers de-quantizes to float32.

        Returns:
            torch.nn.Module: The model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = _GGUF_FILES[self._variant]
        if self.tokenizer is None:
            self._load_tokenizer()
        model_kwargs = {"use_cache": False, "gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
    ):
        """Load and return sample inputs for the AIDC-LLM-Laos model.

        Returns:
            list: [input_ids, attention_mask] tensors that can be fed to the model.
        """
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer()
        input_prompt = [
            {
                "role": "user",
                "content": prompt or self.sample_text,
            }
        ]
        input_text = self.tokenizer.apply_chat_template(
            input_prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return [input_ids, attn_mask]
