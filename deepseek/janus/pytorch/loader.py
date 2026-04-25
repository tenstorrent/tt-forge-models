# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Janus model loader implementation for multimodal understanding.
"""

from typing import Optional

import torch
import janus.models  # registers multi_modality config/model with transformers auto classes
from janus.models import VLChatProcessor
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

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
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available DeepSeek Janus model variants."""

    JANUS_1_3B = "Janus_1_3B"


class ModelLoader(ForgeModel):
    """DeepSeek Janus model loader for multimodal understanding."""

    _VARIANTS = {
        ModelVariant.JANUS_1_3B: ModelConfig(
            pretrained_model_name="deepseek-ai/Janus-1.3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JANUS_1_3B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Janus model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Janus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = VLChatProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Janus model instance."""
        model_name = self._variant_config.pretrained_model_name

        config = AutoConfig.from_pretrained(str(model_name))
        config.language_config._attn_implementation = "eager"

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.2+ always uses torch.device("meta") during __init__,
        # but janus VisionTransformer calls .item() on a linspace tensor in its
        # __init__, which fails on meta tensors. Patch get_init_context to
        # strip the meta device context for this model family.
        #
        # Additionally, patch _adjust_tied_keys_with_tied_pointers to handle
        # models that do not call post_init() and lack all_tied_weights_keys.
        _orig_get_init_context = PreTrainedModel.get_init_context.__func__
        _orig_adjust_tied = PreTrainedModel._adjust_tied_keys_with_tied_pointers

        @classmethod
        def _no_meta_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            return [
                ctx
                for ctx in _orig_get_init_context(
                    cls, dtype, is_quantized, _is_ds_init_called
                )
                if not (isinstance(ctx, torch.device) and ctx.type == "meta")
            ]

        def _safe_adjust_tied(self, missing_keys):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return _orig_adjust_tied(self, missing_keys)

        PreTrainedModel.get_init_context = _no_meta_get_init_context
        PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust_tied
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_name), **model_kwargs
            )
        finally:
            PreTrainedModel.get_init_context = classmethod(_orig_get_init_context)
            PreTrainedModel._adjust_tied_keys_with_tied_pointers = _orig_adjust_tied

        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Janus."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{self.sample_text}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        batch = self.processor(
            conversations=conversation, images=[image], force_batchify=True
        )

        # BatchedVLChatProcessorOutput is a DictOutput (no __iter__); extract
        # only tensor fields into a plain dict so downstream code can use **inputs.
        inputs = {
            key: val
            for key, val in vars(batch).items()
            if isinstance(val, torch.Tensor)
        }

        if dtype_override:
            inputs = {
                key: cast_input_to_type(val, dtype_override)
                for key, val in inputs.items()
            }

        return inputs
