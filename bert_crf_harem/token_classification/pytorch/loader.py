# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT-CRF model loader implementation for token classification (Portuguese NER).
"""

import torch
import torchcrf
from transformers import AutoModelForTokenClassification, BertTokenizer, PreTrainedModel
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


class ModelVariant(StrEnum):
    """Available BERT-CRF HAREM model variants for token classification."""

    HCAERYKS_BERT_CRF_HAREM = "hcaeryks/bert-crf-harem"


class ModelLoader(ForgeModel):
    """BERT-CRF HAREM model loader implementation for token classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.HCAERYKS_BERT_CRF_HAREM: LLMModelConfig(
            pretrained_model_name="hcaeryks/bert-crf-harem",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.HCAERYKS_BERT_CRF_HAREM

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.sample_text = "O presidente Lula visitou a cidade de São Paulo no Brasil"
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="BERT-CRF-HAREM",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load BERT-CRF model for token classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BERT-CRF model instance.
        """

        # hcaeryks/bert-crf-harem has no tokenizer files; the underlying BERT is
        # neuralmind/bert-large-portuguese-cased whose vocab.txt is bundled with
        # the repo and loadable via BertTokenizer directly.
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace (trust_remote_code required for CRF head)
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # The custom BERT_CRF model (from arubenruben/PT-BERT-Large-CRF-HAREM-Default)
        # was written for transformers 4.29.1 and has three incompatibilities with 5.x:
        # 1. It calls BertModel.from_pretrained() inside __init__, which raises RuntimeError
        #    when the meta-device context is active.
        # 2. It doesn't call self.post_init(), so all_tied_weights_keys (required by
        #    _finalize_model_loading in 5.x) is never set.
        # 3. The outer BERT_CRF from_pretrained replaces all non-persistent buffers (including
        #    position_ids properly set by the inner BertModel.from_pretrained) with torch.empty_like
        #    (uninitialized) tensors, and then skips re-init because _is_hf_initialized is already
        #    True. Fix: only replace buffers that are actually on the meta device.
        original_get_init_context = PreTrainedModel.get_init_context.__func__
        original_adjust_tied = PreTrainedModel._adjust_tied_keys_with_tied_pointers
        original_move_missing = PreTrainedModel._move_missing_keys_from_meta_to_device

        @classmethod
        def patched_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            contexts = original_get_init_context(cls, dtype, is_quantized, _is_ds_init_called)
            return [ctx for ctx in contexts if not isinstance(ctx, torch.device)]

        def patched_adjust_tied(self, missing_keys):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return original_adjust_tied(self, missing_keys)

        def patched_move_missing(self, missing_keys, device_map, device_mesh, hf_quantizer):
            # Only replace non-persistent buffers that are still on the meta device;
            # skip ones already on CPU (properly initialized by a nested from_pretrained).
            original_named_non_persistent = self.named_non_persistent_buffers

            def filtered_named_non_persistent():
                for key, buf in original_named_non_persistent():
                    if buf.device.type == "meta":
                        yield key, buf

            self.named_non_persistent_buffers = filtered_named_non_persistent
            try:
                original_move_missing(self, missing_keys, device_map, device_mesh, hf_quantizer)
            finally:
                self.named_non_persistent_buffers = original_named_non_persistent

        PreTrainedModel.get_init_context = patched_get_init_context
        PreTrainedModel._adjust_tied_keys_with_tied_pointers = patched_adjust_tied
        PreTrainedModel._move_missing_keys_from_meta_to_device = patched_move_missing
        try:
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, trust_remote_code=True, **model_kwargs
            )
        finally:
            PreTrainedModel.get_init_context = classmethod(original_get_init_context)
            PreTrainedModel._adjust_tied_keys_with_tied_pointers = original_adjust_tied
            PreTrainedModel._move_missing_keys_from_meta_to_device = original_move_missing

        # BertModel is loaded in float32 (from neuralmind checkpoint) while the outer
        # from_pretrained context sets bfloat16 default dtype, leaving self.linear in
        # bfloat16. Cast everything to dtype_override for consistency.
        if dtype_override is not None:
            model = model.to(dtype_override)

        # pytorch-crf 0.7.2 creates uint8 default masks in CRF.decode and uses them
        # in torch.where inside _viterbi_decode. In PyTorch 2.x torch.where requires
        # boolean predicates; uint8 causes TorchRuntimeError during torch.compile.
        # Patch _viterbi_decode to cast mask to bool before the torch.where calls.
        _orig_viterbi = torchcrf.CRF._viterbi_decode

        def _patched_viterbi(self_crf, emissions, mask):
            return _orig_viterbi(self_crf, emissions, mask.bool())

        torchcrf.CRF._viterbi_decode = _patched_viterbi

        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for BERT-CRF token classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # BERT_CRF forward requires labels and labels_mask arguments
        inputs["labels"] = None
        inputs["labels_mask"] = inputs["attention_mask"].bool()

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for token classification.

        Args:
            co_out: Model output
        """
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )
        predicted_tokens_classes = [
            self.model.config.id2label[t.item()] for t in predicted_token_class_ids
        ]

        print(f"Context: {self.sample_text}")
        print(f"Answer: {predicted_tokens_classes}")
