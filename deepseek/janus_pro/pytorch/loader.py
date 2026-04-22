# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Janus-Pro model loader implementation for multimodal understanding.
"""

from typing import Optional

import torch
from PIL import Image
from janus.models import (
    MultiModalityCausalLM,
    VLChatProcessor,
)  # registers multi_modality with transformers
from janus.models import siglip_vit as _janus_siglip_vit

# Transformers 5.x initializes models under torch.device("meta"), but janus's
# VisionTransformer.__init__ calls .item() on torch.linspace output which fails
# on meta tensors. Patch linspace to always produce CPU tensors in that __init__.
_orig_vit_init = _janus_siglip_vit.VisionTransformer.__init__
_orig_torch_linspace = torch.linspace


def _patched_vit_init(self, *args, **kwargs):
    torch.linspace = lambda *a, **kw: _orig_torch_linspace(
        *a, **{**kw, "device": "cpu"}
    )
    try:
        _orig_vit_init(self, *args, **kwargs)
    finally:
        torch.linspace = _orig_torch_linspace


_janus_siglip_vit.VisionTransformer.__init__ = _patched_vit_init

# Janus's MultiModalityCausalLM doesn't call post_init(), which is required in
# transformers 5.x to set all_tied_weights_keys and other attributes used during
# _finalize_model_loading. Patch __init__ to call post_init() after construction.
from janus.models.modeling_vlm import MultiModalityCausalLM as _MultiModalityCausalLM

_orig_multi_modality_init = _MultiModalityCausalLM.__init__


def _patched_multi_modality_init(self, config, *args, **kwargs):
    _orig_multi_modality_init(self, config, *args, **kwargs)
    if not hasattr(self, "all_tied_weights_keys"):
        self.post_init()


_MultiModalityCausalLM.__init__ = _patched_multi_modality_init

from transformers import AutoModelForCausalLM

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
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available DeepSeek Janus-Pro model variants."""

    JANUS_PRO_1B = "Janus_Pro_1B"


class _JanusProWrapper(torch.nn.Module):
    """Wraps MultiModalityCausalLM to expose a standard forward() interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, pixel_values, attention_mask, images_seq_mask, images_emb_mask
    ):
        inputs_embeds = self.model.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        return self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )


class ModelLoader(ForgeModel):
    """DeepSeek Janus-Pro model loader for multimodal understanding."""

    _VARIANTS = {
        ModelVariant.JANUS_PRO_1B: ModelConfig(
            pretrained_model_name="deepseek-ai/Janus-Pro-1B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JANUS_PRO_1B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Janus-Pro model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="JanusPro",
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
        """Load and return the Janus-Pro model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(str(model_name), **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return _JanusProWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Janus-Pro."""
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

        prepare = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        )

        inputs = {
            "input_ids": prepare.input_ids,
            "pixel_values": prepare.pixel_values,
            "attention_mask": prepare.attention_mask,
            "images_seq_mask": prepare.images_seq_mask,
            "images_emb_mask": prepare.images_emb_mask,
        }

        if dtype_override:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
