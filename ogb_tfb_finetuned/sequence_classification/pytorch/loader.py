# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OGB TFB Finetuned model loader implementation for multi-label sequence classification.

Supports the yangheng/ogb_tfb_finetuned checkpoint, an OmniGenome-based
transformer fine-tuned on the Oxford Genomics Benchmark transcription factor
binding task (919 binary labels) for DNA sequence classification.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

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


def _patch_pretrained_model_for_omnigenome():
    """Lazily initialize all_tied_weights_keys when it is missing.

    OmniGenome's remote code calls init_weights() instead of post_init(), so
    all_tied_weights_keys (normally set by post_init) is never populated.
    transformers v5 accesses it in _adjust_tied_keys_with_tied_pointers.
    """
    if getattr(PreTrainedModel, "_omnigenome_v5_patched", False):
        return
    _orig = PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def _patched(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
                all_submodels=False
            )
        return _orig(self, *args, **kwargs)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched
    PreTrainedModel._omnigenome_v5_patched = True


class ModelVariant(StrEnum):
    """Available OGB TFB Finetuned model variants."""

    OGB_TFB_FINETUNED = "yangheng/ogb_tfb_finetuned"


class ModelLoader(ForgeModel):
    """OGB TFB Finetuned model loader for multi-label DNA sequence classification."""

    _VARIANTS = {
        ModelVariant.OGB_TFB_FINETUNED: ModelConfig(
            pretrained_model_name="yangheng/ogb_tfb_finetuned",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OGB_TFB_FINETUNED

    sample_text = "ACGTACGTACGTACGTACGTACGTACGTACGT"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="OGB TFB Finetuned",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        # OmniGenomeConfig omits several attrs that transformers v5 no longer sets
        # in PretrainedConfig defaults; supply them explicitly.
        for attr, default in [
            ("is_decoder", False),
            ("add_cross_attention", False),
            ("chunk_size_feed_forward", 0),
        ]:
            if not hasattr(config, attr):
                setattr(config, attr, default)

        _patch_pretrained_model_for_omnigenome()

        model = AutoModelForSequenceClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            config=config,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            # Ensure all parameters are in the target dtype, including randomly
            # initialized weights that were missing from the checkpoint due to key
            # mismatch (checkpoint uses "model.encoder.*" but model expects
            # "OmniGenome.encoder.*").
            model.to(dtype_override)

        # Cast attention_mask to hidden_states dtype inside OmniGenomeSelfAttention
        # so that attention_probs stays in the same dtype as value_layer.
        # Without this, the float32 extended attention mask causes attention_scores
        # to upcast to float32, breaking the tt_torch matmul dtype check.
        for mod in model.modules():
            cls = type(mod)
            if cls.__name__ == "OmniGenomeSelfAttention" and not getattr(
                cls, "_dtype_patched", False
            ):
                _orig_attn_fwd = cls.forward

                def _patched_attn_fwd(
                    self,
                    hidden_states,
                    attention_mask=None,
                    *args,
                    _orig=_orig_attn_fwd,
                    **kw,
                ):
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.dtype)
                    return _orig(self, hidden_states, attention_mask, *args, **kw)

                cls.forward = _patched_attn_fwd
                cls._dtype_patched = True
                break

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
