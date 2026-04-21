# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BabyLM Flamingo model loader implementation for multimodal causal language modeling.
"""

from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, ViTImageProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from transformers.models.opt.modeling_opt import OPTDecoder

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...tools.utils import cast_input_to_type


def _update_causal_mask(
    self,
    attention_mask,
    input_tensor,
    cache_position,
    past_key_values,
    output_attentions,
):
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    dtype = input_tensor.dtype
    input_shape = input_tensor.shape[:2]
    return _create_4d_causal_attention_mask(
        input_shape,
        dtype=dtype,
        device=input_tensor.device,
        past_key_values_length=past_seen_tokens,
    )


if not hasattr(OPTDecoder, "_update_causal_mask"):
    OPTDecoder._update_causal_mask = _update_causal_mask


class ModelVariant(StrEnum):
    """Available BabyLM Flamingo model variants."""

    MULTIMODAL_BASELINE = "multimodal_baseline"


class ModelLoader(ForgeModel):
    """BabyLM Flamingo model loader for multimodal causal language modeling."""

    _VARIANTS = {
        ModelVariant.MULTIMODAL_BASELINE: ModelConfig(
            pretrained_model_name="BabyLM-community/babylm-multimodal-baseline-flamingo",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTIMODAL_BASELINE

    sample_text = "A photo of"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize BabyLM Flamingo model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BabyLM Flamingo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        model_name = self._variant_config.pretrained_model_name
        FlamingoProcessor = get_class_from_dynamic_module(
            "processor_flamingo.FlamingoProcessor",
            model_name,
            trust_remote_code=True,
        )
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = FlamingoProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BabyLM Flamingo model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **kwargs,
        )
        model.config.use_cache = False
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for BabyLM Flamingo."""
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(
            images=image,
            text=self.sample_text,
            return_tensors="pt",
        )

        if dtype_override:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return dict(inputs)
