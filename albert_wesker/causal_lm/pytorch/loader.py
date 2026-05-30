# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Albert_Wesker-1B model loader implementation for causal language modeling.

Albert_Wesker-1B is a Gemma 3 (1B) fine-tune distributed only in GGUF format
(mradermacher/Albert_Wesker-1B-GGUF). There is no separate safetensors base
repo, so the config, tokenizer and weights are all reconstructed directly from
the GGUF checkpoint via the ``gguf_file`` argument to ``from_pretrained``.
A K-quant file (Q4_K_M) is used; transformers dequantizes it to the requested
torch dtype on load (IQ/imatrix quants are intentionally avoided as the
transformers GGUF loader does not support them).
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
    """Available Albert_Wesker model variants for causal LM."""

    ALBERT_WESKER_1B = "1B"


class ModelLoader(ForgeModel):
    """Albert_Wesker-1B model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.ALBERT_WESKER_1B: LLMModelConfig(
            pretrained_model_name="mradermacher/Albert_Wesker-1B-GGUF",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALBERT_WESKER_1B

    # GGUF checkpoint to load config / tokenizer / weights from (per variant).
    _GGUF_FILES = {
        ModelVariant.ALBERT_WESKER_1B: "Albert_Wesker-1B.Q4_K_M.gguf",
    }

    # A longer, fully-attended prompt keeps every compared logit a real token,
    # which avoids the short-sequence bf16 accumulation error that otherwise
    # drags down on-device PCC for deep decoders.
    sample_text = (
        "Albert Wesker stood at the edge of the abandoned laboratory, the "
        "flickering lights casting long shadows across the cold steel floor. "
        "He adjusted his sunglasses and considered his next move carefully, "
        "weighing every possible outcome before he spoke a single word."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Albert_Wesker 1B",
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
        gguf_file = self._GGUF_FILES[self._variant]
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Albert_Wesker-1B causal_lm model instance.

        Args:
            dtype_override: Optional torch dtype to load the (dequantized) weights as.
                If not provided, transformers uses its default dtype.

        Returns:
            torch.nn.Module: The Albert_Wesker-1B model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]
        if self.tokenizer is None:
            self._load_tokenizer()
        model_kwargs = {"use_cache": False, "gguf_file": gguf_file}
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
        """Load and return sample inputs for the Albert_Wesker-1B model.

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
