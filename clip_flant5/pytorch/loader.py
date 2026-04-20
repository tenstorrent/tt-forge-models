# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLIP-FlanT5 model loader implementation for visual question answering.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available CLIP-FlanT5 model variants."""

    XL = "XL"
    XXL = "XXL"


class ModelLoader(ForgeModel):
    """CLIP-FlanT5 model loader implementation for visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.XL: ModelConfig(
            pretrained_model_name="zhiqiulin/clip-flant5-xl",
        ),
        ModelVariant.XXL: ModelConfig(
            pretrained_model_name="zhiqiulin/clip-flant5-xxl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CLIP_FlanT5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_processor()

        question = "Does this image show 'two cats on a couch'? Answer yes or no."
        text_inputs = self.tokenizer(question, return_tensors="pt", padding=True)

        decoder_input_ids = torch.full(
            (1, 1), self.tokenizer.pad_token_id, dtype=torch.long
        )

        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
        }

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.tokenizer is None:
            self._load_processor()

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
