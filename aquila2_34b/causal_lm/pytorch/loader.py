# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Aquila2-34B model loader implementation for causal language modeling.

Loads the BAAI/Aquila2-34B model, a 34B parameter bilingual (English/Chinese)
base causal language model from BAAI. The architecture ships as a custom
HuggingFace implementation (``modeling_aquila.py``/``configuration_aquila.py``),
so loading requires ``trust_remote_code=True``.

Available variants:
- AQUILA2_34B: BAAI/Aquila2-34B
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

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
    """Available Aquila2 model variants for causal language modeling."""

    AQUILA2_34B = "34B"


class ModelLoader(ForgeModel):
    """Aquila2-34B model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AQUILA2_34B: LLMModelConfig(
            pretrained_model_name="BAAI/Aquila2-34B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AQUILA2_34B

    sample_text = "Hey how are you doing today?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Aquila2",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        # Newer transformers populate a computed rope_scaling dict like
        # {"rope_type": "default", ...} even when the raw config has None.
        # Aquila's custom modeling code keys off rope_scaling["type"], so
        # normalize to None to match the original config behavior.
        if (
            config.rope_scaling
            and config.rope_scaling.get("rope_type") == "default"
            and "type" not in config.rope_scaling
        ):
            config.rope_scaling = None

        model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_token_ids = logits.argmax(dim=-1)
        return self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
