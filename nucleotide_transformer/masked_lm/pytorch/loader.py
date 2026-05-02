# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nucleotide Transformer v2 model loader implementation for masked language modeling.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


def _patch_transformers_nucleotide_transformer():
    """Apply transformers 5.x compatibility shims for the Nucleotide Transformer custom model code.

    The HuggingFace repo ships custom modeling_esm.py and esm_config.py that were written against
    transformers 4.x and are incompatible with transformers 5.x in several ways:

    1. find_pruneable_heads_and_indices was removed from transformers.pytorch_utils
    2. PretrainedConfig no longer sets is_decoder/chunk_size_feed_forward/add_cross_attention defaults
    3. all_tied_weights_keys is not initialized by the custom EsmForMaskedLM (transformers 5.x new requirement)
    4. PreTrainedModel.get_head_mask was removed in transformers 5.x
    """
    import transformers.pytorch_utils as _pu
    import transformers.modeling_utils as _mu

    # 1. Shim: find_pruneable_heads_and_indices removed from transformers.pytorch_utils in 5.x
    if not hasattr(_pu, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(
            heads, n_heads, head_size, already_pruned_heads
        ):
            mask = torch.ones(n_heads, head_size)
            heads_set = set(heads) - already_pruned_heads
            for head in heads_set:
                h = head - sum(1 if h2 < head else 0 for h2 in already_pruned_heads)
                mask[h] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads_set, index

        _pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # 2. Shim: get_head_mask removed from PreTrainedModel in transformers 5.x
    if not hasattr(_mu.PreTrainedModel, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        _mu.PreTrainedModel.get_head_mask = get_head_mask

    # 3. Shim: all_tied_weights_keys not initialized by custom EsmForMaskedLM under transformers 5.x
    #    meta-device init context. Patch _adjust_tied_keys_with_tied_pointers to handle missing attr.
    _orig_adjust = _mu.PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def _patched_adjust(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            object.__setattr__(self, "all_tied_weights_keys", {})
        return _orig_adjust(self, *args, **kwargs)

    _mu.PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched_adjust


class ModelVariant(StrEnum):
    """Available Nucleotide Transformer model variants for masked language modeling."""

    V2_50M_MULTI_SPECIES = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    V2_250M_MULTI_SPECIES = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"


class ModelLoader(ForgeModel):
    """Nucleotide Transformer v2 model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.V2_50M_MULTI_SPECIES: LLMModelConfig(
            pretrained_model_name="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            max_length=128,
        ),
        ModelVariant.V2_250M_MULTI_SPECIES: LLMModelConfig(
            pretrained_model_name="InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_50M_MULTI_SPECIES

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nucleotide Transformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        _patch_transformers_nucleotide_transformer()

        model_name = self._variant_config.pretrained_model_name
        config = self.load_config()
        # Defaults removed from PretrainedConfig in transformers 5.x that custom esm_config.py relies on
        if not hasattr(config, "is_decoder"):
            config.is_decoder = False
        if not hasattr(config, "chunk_size_feed_forward"):
            config.chunk_size_feed_forward = 0
        if not hasattr(config, "add_cross_attention"):
            config.add_cross_attention = False

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Sample DNA sequence
        dna_sequence = "ATTCCGATTCCGATTCCG"

        max_length = self._variant_config.max_length
        tokens_ids = self.tokenizer(
            dna_sequence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return tokens_ids

    def decode_output(self, outputs, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        # Get predicted tokens
        predicted_token_ids = torch.argmax(logits, dim=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_ids[0])
        return predicted_tokens

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
