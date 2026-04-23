# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ModChemBERT model loader implementation for masked language modeling.

ModChemBERT is a ModernBERT-based chemical language model pretrained on SMILES
strings for molecular information retrieval and related downstream tasks.

Reference: https://huggingface.co/Derify/ModChemBERT-IR-BASE
"""

from typing import Optional

import torch
import transformers.models.modernbert.modeling_modernbert as _mbert_module
from transformers import AutoModelForMaskedLM, AutoTokenizer

# transformers>=5.x renamed MODERNBERT_ATTENTION_FUNCTION to ALL_ATTENTION_FUNCTIONS
# and removed _pad_modernbert_output / _unpad_modernbert_input (flash_attention_2 only helpers)
if not hasattr(_mbert_module, "MODERNBERT_ATTENTION_FUNCTION"):
    _mbert_module.MODERNBERT_ATTENTION_FUNCTION = _mbert_module.ALL_ATTENTION_FUNCTIONS

if not hasattr(_mbert_module, "_unpad_modernbert_input"):

    def _unpad_modernbert_input(inputs, attention_mask):
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen = int(seqlens.max().item())
        cu_seqlens = torch.cat([seqlens.new_zeros(1), seqlens.cumsum(dim=0)])
        return inputs.flatten(0, 1)[indices], indices, cu_seqlens, max_seqlen

    _mbert_module._unpad_modernbert_input = _unpad_modernbert_input

if not hasattr(_mbert_module, "_pad_modernbert_output"):

    def _pad_modernbert_output(inputs, indices, batch, seqlen):
        output = inputs.new_zeros(batch * seqlen, *inputs.shape[1:])
        output[indices] = inputs
        return output.view(batch, seqlen, *inputs.shape[1:])

    _mbert_module._pad_modernbert_output = _pad_modernbert_output

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ModChemBERT model variants for masked language modeling."""

    IR_BASE = "IR-BASE"


class ModelLoader(ForgeModel):
    """ModChemBERT model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.IR_BASE: ModelConfig(
            pretrained_model_name="Derify/ModChemBERT-IR-BASE",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IR_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ModChemBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        # SMILES string (benzene) with a masked substituent position
        test_input = "c1ccccc1[MASK]"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs):
        if self.tokenizer is None:
            self._load_tokenizer()

        if isinstance(outputs, list):
            logits = outputs[0].logits if hasattr(outputs[0], "logits") else outputs[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        inputs = self.load_inputs()

        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]

        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

        output = self.tokenizer.decode(predicted_token_id)

        return f"Output: {output}"
