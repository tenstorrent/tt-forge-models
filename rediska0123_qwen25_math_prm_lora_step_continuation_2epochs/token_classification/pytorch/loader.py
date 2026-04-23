# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
rediska0123 Qwen2.5-Math-PRM LoRA step-continuation (2 epochs) model loader
implementation for process-reward token classification.
"""
from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache

if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = (
        lambda self, new_seq_len=None, layer_idx=None: self.get_seq_length(layer_idx)
    )
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
    """Available rediska0123 Qwen2.5-Math-PRM LoRA step-continuation variants."""

    STEP_CONTINUATION_2EPOCHS = "step_continuation_2epochs"


class ModelLoader(ForgeModel):
    """rediska0123 Qwen2.5-Math-PRM LoRA step-continuation model loader for
    process reward token classification tasks.
    """

    _VARIANTS = {
        ModelVariant.STEP_CONTINUATION_2EPOCHS: LLMModelConfig(
            pretrained_model_name="rediska0123/qwen25-math-prm-lora-step-continuation-2epochs",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STEP_CONTINUATION_2EPOCHS

    sample_text = (
        "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast "
        "every morning and bakes muffins for her friends every day with four. "
        "She sells the remainder at the farmers' market daily for $2 per fresh "
        "duck egg. How much in dollars does she make every day at the farmers' "
        "market?\n"
        "Step 1: Janet\u2019s ducks lay 16 eggs per day.\n"
        "Step 2: She eats 3 and bakes 4, so she uses 3 + 4 = 7 eggs.\n"
        "Step 3: She sells 16 - 7 = 9 eggs at $2 each.\n"
        "Step 4: She makes 9 * $2 = $18 per day."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="rediska0123-qwen25-math-prm-lora-step-continuation",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **tokenizer_kwargs
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
        config.pad_token_id = self.tokenizer.pad_token_id

        model = AutoModel.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, inputs=None):
        logits = outputs[0]
        predicted_class_ids = logits.argmax(-1)
        return predicted_class_ids.tolist()
