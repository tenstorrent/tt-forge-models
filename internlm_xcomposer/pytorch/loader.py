# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternLM-XComposer model loader implementation for multimodal visual question answering.
"""

from typing import Optional

import torch
import torch.nn as nn
import transformers.modeling_utils
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer

from ...base import ForgeModel


def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


transformers.modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
transformers.modeling_utils.find_pruneable_heads_and_indices = (
    _find_pruneable_heads_and_indices
)
transformers.modeling_utils.prune_linear_layer = prune_linear_layer

if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
    transformers.PreTrainedModel.all_tied_weights_keys = {}
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class InternLMXComposerWrapper(nn.Module):
    def __init__(self, internlm_model):
        super().__init__()
        self.internlm_model = internlm_model

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.internlm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits


class ModelVariant(StrEnum):
    """Available InternLM-XComposer model variants."""

    INTERNLM_XCOMPOSER_7B = "7B"


class ModelLoader(ForgeModel):
    """InternLM-XComposer model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.INTERNLM_XCOMPOSER_7B: ModelConfig(
            pretrained_model_name="internlm/internlm-xcomposer-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERNLM_XCOMPOSER_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self._full_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="InternLM-XComposer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {
            "trust_remote_code": True,
        }
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "max_length"):
            config.max_length = config.max_position_embeddings
        config.device = "cpu"
        model_kwargs["config"] = config

        full_model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        full_model.eval()
        full_model.tokenizer = self.tokenizer
        self._full_model = full_model

        wrapper = InternLMXComposerWrapper(full_model.internlm_model)
        wrapper.eval()
        self.model = wrapper

        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        text = "<|User|>: Please describe this image in detail.\n<|Bot|>:"
        texts = [text] * batch_size

        with torch.no_grad():
            tokens = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self._full_model.max_length,
                add_special_tokens=False,
            )
            inputs_embeds = self._full_model.internlm_model.model.embed_tokens(
                tokens.input_ids
            )
            attention_mask = tokens.attention_mask

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

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
            logits = outputs if torch.is_tensor(outputs) else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)
