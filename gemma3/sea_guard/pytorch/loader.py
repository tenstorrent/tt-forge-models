# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma SEA-Guard 12B model loader (GGUF-quantized) for causal language modeling.

The weights are distributed only as GGUF files (mradermacher/Gemma-SEA-Guard-12B-2602-GGUF),
a quantization of aisingapore/Gemma-SEA-Guard-12B-2602 (Gemma 3 text architecture).
transformers dequantizes the GGUF back to floating point and instantiates a normal
``Gemma3ForCausalLM`` module; the tokenizer is embedded in the GGUF as well.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
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
    """Available Gemma SEA-Guard model variants."""

    GEMMA_SEA_GUARD_12B_Q4_K_M = "12b_q4_k_m"


class ModelLoader(ForgeModel):
    """Gemma SEA-Guard 12B (GGUF) loader for causal language modeling tasks."""

    # GGUF repo and the specific quant file to load.
    _GGUF_REPO = "mradermacher/Gemma-SEA-Guard-12B-2602-GGUF"

    _VARIANTS = {
        ModelVariant.GEMMA_SEA_GUARD_12B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Gemma-SEA-Guard-12B-2602-GGUF",
            max_length=256,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.GEMMA_SEA_GUARD_12B_Q4_K_M: "Gemma-SEA-Guard-12B-2602.Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_SEA_GUARD_12B_Q4_K_M

    sample_text = "I want to learn how to cook a delicious meal for my family."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Gemma SEA-Guard 12B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self):
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer (embedded in the GGUF) for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma SEA-Guard model instance.

        The GGUF checkpoint is dequantized by transformers into a standard
        ``Gemma3ForCausalLM`` module.

        Args:
            dtype_override: Optional torch dtype to cast the model to. If not
                provided, the model uses the dtype produced by GGUF dequantization.

        Returns:
            torch.nn.Module: The Gemma SEA-Guard model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"use_cache": False, "gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

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
        prompt: Optional[str] = None,
    ):
        """Load and return sample inputs for the Gemma SEA-Guard model.

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
