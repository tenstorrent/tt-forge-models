# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Yanis-Gerst/fine_tune model loader implementation for image-text-to-text tasks.

Fine-tuned from microsoft/Phi-4-multimodal-instruct.
"""

import torch
import transformers.modeling_utils as _modeling_utils
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from transformers.image_utils import load_image
from typing import Optional

# SlidingWindowCache was removed in transformers 5.x; patch for model remote code compatibility
try:
    from transformers.cache_utils import SlidingWindowCache as _  # noqa: F401
except ImportError:
    import transformers.cache_utils as _cache_utils
    from transformers.cache_utils import StaticCache

    class _SlidingWindowCacheStub(StaticCache):
        pass

    _cache_utils.SlidingWindowCache = _SlidingWindowCacheStub

# peft 0.19+ requires prepare_inputs_for_generation on the inner model when task_type=CAUSAL_LM.
# The model wraps the inner Phi4MMModel (which lacks this method); patch PeftModelForCausalLM to
# tolerate the missing attribute since generation is handled by the outer Phi4MMForCausalLM.
import peft as _peft


def _peft_causal_lm_init_compat(
    self, model, peft_config, adapter_name="default", **kwargs
):
    super(_peft.PeftModelForCausalLM, self).__init__(
        model, peft_config, adapter_name, **kwargs
    )
    self.base_model_prepare_inputs_for_generation = getattr(
        self.base_model, "prepare_inputs_for_generation", None
    )


_peft.PeftModelForCausalLM.__init__ = _peft_causal_lm_init_compat

# transformers 5.x expects _tied_weights_keys as a dict; old model code uses a list.
# Patch get_expanded_tied_weights_keys to convert list format to empty dict so loading works.
_orig_get_expanded = _modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys


def _get_expanded_tied_weights_keys_compat(self, all_submodels=False):
    if isinstance(self._tied_weights_keys, list):
        _orig_keys = self._tied_weights_keys
        self._tied_weights_keys = {}
        try:
            return _orig_get_expanded(self, all_submodels=all_submodels)
        finally:
            self._tied_weights_keys = _orig_keys
    return _orig_get_expanded(self, all_submodels=all_submodels)


_modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = (
    _get_expanded_tied_weights_keys_compat
)

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Yanis-Gerst/fine_tune model variants for image-text-to-text tasks."""

    FINE_TUNE = "fine_tune"


class ModelLoader(ForgeModel):
    """Yanis-Gerst/fine_tune model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.FINE_TUNE: LLMModelConfig(
            pretrained_model_name="Yanis-Gerst/fine_tune",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FINE_TUNE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Yanis-Gerst/fine_tune",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            trust_remote_code=True,
            **kwargs,
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # NemoConvSubsampling calls int() on tensors during __init__, which fails with
        # the meta device context that transformers 5.x always uses. Patch get_init_context
        # to exclude meta device so concrete tensor values are available during init.
        _orig_get_init_context = (
            _modeling_utils.PreTrainedModel.get_init_context.__func__
        )

        @classmethod  # type: ignore[misc]
        def _get_init_context_no_meta(cls, dtype, is_quantized, _is_ds_init_called):
            contexts = _orig_get_init_context(
                cls, dtype, is_quantized, _is_ds_init_called
            )
            return [
                c
                for c in contexts
                if not (isinstance(c, torch.device) and c.type == "meta")
            ]

        _modeling_utils.PreTrainedModel.get_init_context = _get_init_context_no_meta
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _modeling_utils.PreTrainedModel.get_init_context = classmethod(
                _orig_get_init_context
            )

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        )

        prompt = "<|user|><|image_1|>\nWhat is in this image?<|end|><|assistant|>"
        inputs = self.processor(prompt, images=[image], return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
