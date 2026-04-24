# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XLM-RoBERTa model loader implementation for masked language modeling (PyTorch).
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available XLM-RoBERTa For Masked LM model variants."""

    TF_XLM_ROBERTA_BASE = "Tf_Xlm_Roberta_Base"
    TWITTER_XLM_ROBERTA_BASE = "Twitter_Xlm_Roberta_Base"
    NOMIC_XLM_2048 = "Nomic_Xlm_2048"


class ModelLoader(ForgeModel):
    """XLM-RoBERTa model loader implementation for masked language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TF_XLM_ROBERTA_BASE: ModelConfig(
            pretrained_model_name="jplu/tf-xlm-roberta-base",
        ),
        ModelVariant.TWITTER_XLM_ROBERTA_BASE: ModelConfig(
            pretrained_model_name="cardiffnlp/twitter-xlm-roberta-base",
        ),
        ModelVariant.NOMIC_XLM_2048: ModelConfig(
            pretrained_model_name="nomic-ai/nomic-xlm-2048",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TF_XLM_ROBERTA_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="XLM-RoBERTa",
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

    @staticmethod
    def _patch_nomic_bert_remap(model_name):
        """Patch remap_bert_state_dict to handle tied decoder weight absent from state dict.

        nomic-xlm-2048 ties cls.predictions.decoder.weight to word embeddings, so it
        is not stored in the checkpoint. remap_bert_state_dict unconditionally accesses
        that key, causing a KeyError. We temporarily inject the tied weight so the
        function can proceed without error.
        """
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # Loading the class triggers import of the modeling module into sys.modules.
        get_class_from_dynamic_module(
            "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertForPreTraining",
            model_name,
        )

        for mod_name, mod in list(sys.modules.items()):
            if (
                "nomic" in mod_name
                and "bert" in mod_name
                and hasattr(mod, "remap_bert_state_dict")
            ):
                if getattr(mod, "_remap_patched", False):
                    return
                orig = mod.remap_bert_state_dict

                def _safe_remap(state_dict, config, **kwargs):
                    key = "cls.predictions.decoder.weight"
                    emb_key = "bert.embeddings.word_embeddings.weight"
                    if key not in state_dict and emb_key in state_dict:
                        state_dict = dict(state_dict)
                        state_dict[key] = state_dict[emb_key]
                    return orig(state_dict, config, **kwargs)

                mod.remap_bert_state_dict = _safe_remap
                mod._remap_patched = True
                return

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if self._variant == ModelVariant.TF_XLM_ROBERTA_BASE:
            model_kwargs["from_tf"] = True
        if self._variant == ModelVariant.NOMIC_XLM_2048:
            model_kwargs["trust_remote_code"] = True
            self._patch_nomic_bert_remap(pretrained_model_name)
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        test_input = f"The capital of France is {self.tokenizer.mask_token}."

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
