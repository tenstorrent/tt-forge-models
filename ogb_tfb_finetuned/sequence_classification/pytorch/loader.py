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

        config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        # OmniGenomeConfig omits several attrs that PretrainedConfig used to provide
        # as defaults in older transformers; patch them for compatibility with 5.x
        for attr, default in [
            ("is_decoder", False),
            ("add_cross_attention", False),
            ("chunk_size_feed_forward", 0),
        ]:
            if not hasattr(config, attr):
                setattr(config, attr, default)

        # OmniGenomeForSequenceClassification uses the transformers 4.x pattern of calling
        # init_weights() directly in __init__; transformers 5.x requires post_init() to be
        # called first so that all_tied_weights_keys is set before _finalize_model_loading.
        # Load the model class explicitly so we can patch its __init__ before from_pretrained.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        _seq_cls_path = config.auto_map.get("AutoModelForSequenceClassification", "")
        if _seq_cls_path:
            _cls = get_class_from_dynamic_module(
                _seq_cls_path,
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )
            if not getattr(_cls, "_patched_transformers5", False):
                _orig_init = _cls.__init__

                def _new_init(self_model, cfg, _orig=_orig_init, _c=_cls):
                    _c.init_weights = lambda s: None
                    try:
                        _orig(self_model, cfg)
                    finally:
                        if "init_weights" in _c.__dict__:
                            del _c.init_weights
                    self_model.post_init()

                _cls.__init__ = _new_init
                _cls._patched_transformers5 = True

            # Patch RotaryEmbedding to cast cos/sin to the input dtype so that
            # bfloat16 query/key tensors are not upcast to float32 during rotary
            # position encoding (inv_freq is registered as float32).
            import sys

            _omnigenome_mod_name = _cls.__module__.rsplit(".", 1)[0]
            _rot_mod = sys.modules.get(_omnigenome_mod_name)
            if _rot_mod is None:
                _rot_mod = sys.modules.get(_cls.__module__)
            _rot_cls = getattr(_rot_mod, "RotaryEmbedding", None)
            if _rot_cls is not None and not getattr(_rot_cls, "_patched_dtype", False):
                _orig_update = _rot_cls._update_cos_sin_tables

                def _patched_update(self_rot, x, seq_dimension=2, _orig=_orig_update):
                    cos, sin = _orig(self_rot, x, seq_dimension)
                    return cos.to(x.dtype), sin.to(x.dtype)

                _rot_cls._update_cos_sin_tables = _patched_update
                _rot_cls._patched_dtype = True

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

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
