# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT MeSH term classification model loader implementation.
"""
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available BERT MeSH classification model variants."""

    MARCMENDEZ_AILY_BERT_MESH_TERMS = "marcmendez_aily_BertMeshTerms"


class ModelLoader(ForgeModel):
    """BERT MeSH term classification model loader implementation."""

    _VARIANTS = {
        ModelVariant.MARCMENDEZ_AILY_BERT_MESH_TERMS: ModelConfig(
            pretrained_model_name="marcmendez-aily/BertMeshTerms",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MARCMENDEZ_AILY_BERT_MESH_TERMS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BERT_MeSH",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # BertMeshModel.__init__ (remote code) calls AutoModel.from_pretrained internally
        # and does not call self.post_init(). Transformers 5.x introduced several
        # breaking changes that need workarounds here:
        #
        # 1. get_init_context unconditionally wraps model __init__ in a meta-device
        #    context. The inner from_pretrained sees the meta device and raises
        #    RuntimeError. Fix: strip the meta-device entry so both loads run on CPU.
        #
        # 2. _finalize_model_loading accesses model.all_tied_weights_keys, which is
        #    set by post_init(). Since BertMeshModel never calls post_init(), the
        #    attribute is missing. Fix: call post_init() on the model before finalization
        #    if the attribute has not been initialised.
        #
        # 3. _move_missing_keys_from_meta_to_device overwrites non-persistent buffers
        #    (e.g. position_ids) with torch.empty_like (garbage). Fix: save and restore
        #    the correctly-seeded buffers after finalization.
        #
        # 4. The inner from_pretrained loads its own dtype context (float32 for BERT),
        #    which causes convert_and_load_state_dict_in_model to preserve float32 for
        #    the bert submodule instead of converting to bfloat16. Fix: explicitly cast
        #    the model to dtype_override after loading.
        original_get_init_context = PreTrainedModel.get_init_context.__func__
        original_finalize = PreTrainedModel._finalize_model_loading

        def _no_meta_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            return [
                c
                for c in original_get_init_context(cls, dtype, is_quantized, _is_ds_init_called)
                if not (isinstance(c, torch.device) and c == torch.device("meta"))
            ]

        def _finalize_with_post_init(model, load_config, loading_info):
            if not hasattr(model, "all_tied_weights_keys"):
                model.post_init()
            # _move_missing_keys_from_meta_to_device replaces every non-persistent
            # buffer with torch.empty_like (uninitialized) even when not on meta
            # device. Save and restore them so correctly-seeded buffers like
            # bert.embeddings.position_ids = arange(512) survive finalization.
            saved = {
                key: buf.clone()
                for key, buf in model.named_non_persistent_buffers()
                if buf is not None and buf.device.type != "meta"
            }
            result = original_finalize(model, load_config, loading_info)
            for key, saved_buf in saved.items():
                parent_name, buf_name = key.rsplit(".", 1) if "." in key else ("", key)
                parent = model.get_submodule(parent_name) if parent_name else model
                parent._buffers[buf_name] = saved_buf
            return result

        PreTrainedModel.get_init_context = classmethod(_no_meta_get_init_context)
        PreTrainedModel._finalize_model_loading = staticmethod(_finalize_with_post_init)
        try:
            model = AutoModel.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                **model_kwargs,
            )
        finally:
            PreTrainedModel.get_init_context = classmethod(original_get_init_context)
            PreTrainedModel._finalize_model_loading = staticmethod(original_finalize)

        if dtype_override is not None:
            model.to(dtype_override)
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        sample_text = "Effect of aspirin on cardiovascular events and bleeding in the healthy elderly."

        inputs = self.tokenizer(
            sample_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        probabilities = torch.sigmoid(logits)
        predicted_indices = (probabilities > 0.5).nonzero(as_tuple=True)[-1]

        if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
            predicted_labels = [
                self.model.config.id2label[idx.item()] for idx in predicted_indices
            ]
        else:
            predicted_labels = [str(idx.item()) for idx in predicted_indices]

        print(f"Predicted MeSH Terms: {predicted_labels}")
