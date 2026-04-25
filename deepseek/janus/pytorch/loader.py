# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Janus model loader implementation for multimodal understanding.
"""

from typing import Optional

import torch
from PIL import Image

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


def _patch_janus():
    """Apply compatibility patches for janus package with transformers 5.x.

    - VisionTransformer.__init__ calls .item() on tensors during meta-device init;
      wrapping in torch.device('cpu') avoids this.
    - MultiModalityCausalLM.__init__ does not call post_init(), which is required
      by transformers 5.x to initialise all_tied_weights_keys.
    - language_config has _attn_implementation='flash_attention_2'; override to
      'eager' to avoid the flash_attn dependency.
    """
    import janus.models.siglip_vit as svit
    from janus.models.modeling_vlm import MultiModalityCausalLM

    if getattr(_patch_janus, "_applied", False):
        return
    _patch_janus._applied = True

    _orig_vit_init = svit.VisionTransformer.__init__

    def _vit_init_cpu(self, *args, **kwargs):
        with torch.device("cpu"):
            _orig_vit_init(self, *args, **kwargs)

    svit.VisionTransformer.__init__ = _vit_init_cpu

    _orig_mlm_init = MultiModalityCausalLM.__init__

    def _mlm_init_with_post_init(self, config):
        if hasattr(config, "language_config") and hasattr(
            config.language_config, "_attn_implementation"
        ):
            config.language_config._attn_implementation = "eager"
        _orig_mlm_init(self, config)
        self.post_init()

    MultiModalityCausalLM.__init__ = _mlm_init_with_post_init


class JanusModelWrapper(torch.nn.Module):
    """Wraps MultiModalityCausalLM to expose a single forward() for tracing."""

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
        from janus.models import VLChatProcessor

        self.processor = VLChatProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Janus model instance."""
        from janus.models import MultiModalityCausalLM

        _patch_janus()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        base_model = MultiModalityCausalLM.from_pretrained(
            str(model_name), **model_kwargs
        )
        base_model.eval()

        if self.processor is None:
            self._load_processor()

        return JanusModelWrapper(base_model)

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

        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        )

        inputs = {
            "input_ids": prepare_inputs.input_ids,
            "pixel_values": prepare_inputs.pixel_values,
            "attention_mask": prepare_inputs.attention_mask,
            "images_seq_mask": prepare_inputs.images_seq_mask,
            "images_emb_mask": prepare_inputs.images_emb_mask,
        }

        if dtype_override:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype_override)

        return inputs
