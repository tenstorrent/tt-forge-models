# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4V model loader implementation for multimodal conditional generation.
"""
from types import MethodType

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _patch_batch_encode_plus(tokenizer):
    """Restore batch_encode_plus removed in transformers v5 for GLM-4V compatibility."""
    if hasattr(tokenizer, "batch_encode_plus"):
        return

    def batch_encode_plus(
        batch_text_or_ids,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        is_split_into_words=False,
        add_special_tokens=False,
        **kwargs,
    ):
        if is_split_into_words:
            encoded = [
                {"input_ids": ids, "attention_mask": [1] * len(ids)}
                for ids in batch_text_or_ids
            ]
            return tokenizer.pad(
                encoded,
                padding=padding,
                max_length=max_length,
                return_tensors=return_tensors,
            )
        return tokenizer(
            batch_text_or_ids,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            is_split_into_words=is_split_into_words,
        )

    tokenizer.batch_encode_plus = batch_encode_plus


def _patch_inplace_ops(model):
    """Replace in-place ops in get_masks with out-of-place equivalents for TT compiler."""
    transformer = model.transformer

    def get_masks(self, input_embeds, past_key_values, padding_mask=None):
        batch_size, seq_length, embed_size = input_embeds.shape
        full_attention_mask = torch.ones(
            batch_size, seq_length, seq_length, device=input_embeds.device
        )
        full_attention_mask = full_attention_mask.tril()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat(
                (
                    torch.ones(
                        batch_size,
                        seq_length,
                        past_length,
                        device=input_embeds.device,
                    ),
                    full_attention_mask,
                ),
                dim=-1,
            )
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask = full_attention_mask - padding_mask.unsqueeze(-1) + 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask = full_attention_mask.unsqueeze(1)
        return full_attention_mask

    transformer.get_masks = MethodType(get_masks, transformer)


class ModelVariant(StrEnum):
    """Available GLM-4V model variants for multimodal conditional generation."""

    GLM_4V_9B = "glm_4v_9b"


class ModelLoader(ForgeModel):
    """GLM-4V model loader implementation for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.GLM_4V_9B: LLMModelConfig(
            pretrained_model_name="zai-org/glm-4v-9b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4V_9B

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="glm_4v",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        _patch_batch_encode_plus(self.tokenizer)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "max_length"):
            config.max_length = config.seq_length
        if not hasattr(config, "use_cache"):
            config.use_cache = True

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        _patch_inplace_ops(model)
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.tokenizer is None:
            self._load_tokenizer()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)
