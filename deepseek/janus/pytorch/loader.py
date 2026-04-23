# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Janus model loader implementation for multimodal understanding.
"""

from typing import Optional

import torch
from PIL import Image
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.modeling_vlm import MultiModalityConfig
from torch.overrides import TorchFunctionMode

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


class _ForceCPUConstructors(TorchFunctionMode):
    # Transformers 5.x initializes models under torch.device("meta") via DeviceContext
    # (a TorchFunctionMode subclass). DeviceContext has higher priority in the mode stack
    # and injects device=meta into all tensor constructors (including torch.linspace).
    # Janus's SigLIP ViT calls torch.linspace(...).item() in __init__, which fails on
    # meta tensors. This mode intercepts after DeviceContext and overrides device=meta
    # back to CPU for constructor calls so that .item() succeeds.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        try:
            from torch.utils._device import _device_constructors

            if func in _device_constructors():
                device = kwargs.get("device")
                if device is not None and torch.device(device).type == "meta":
                    kwargs["device"] = "cpu"
        except Exception:
            pass
        return func(*args, **kwargs)


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

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load config and clear flash_attention_2 from language_config — flash_attn
        # is not installed in this environment and the language_config JSON has it set.
        config = MultiModalityConfig.from_pretrained(model_name, trust_remote_code=True)
        config.language_config._attn_implementation = None

        with _ForceCPUConstructors():
            model = MultiModalityCausalLM.from_pretrained(
                str(model_name), config=config, **model_kwargs
            )
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

        inputs = self.processor(
            conversations=conversation, images=[image], force_batchify=True
        )

        if dtype_override:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs
