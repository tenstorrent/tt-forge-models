# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cambrian-S model loader implementation for multimodal visual question answering.
"""

import torch
from PIL import Image
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Cambrian-S model variants."""

    CAMBRIAN_S_7B_S3 = "S_7B_S3"
    CAMBRIAN_S_3B = "S_3B"


class ModelLoader(ForgeModel):
    """Cambrian-S model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.CAMBRIAN_S_7B_S3: ModelConfig(
            pretrained_model_name="nyu-visionx/Cambrian-S-7B-S3",
        ),
        ModelVariant.CAMBRIAN_S_3B: ModelConfig(
            pretrained_model_name="nyu-visionx/Cambrian-S-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CAMBRIAN_S_7B_S3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Cambrian-S",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_cambrian_init():
        """Patch CambrianQwenForCausalLM.__init__ to avoid clearing rope_parameters.

        In transformers 5.x, config.rope_scaling is a property that delegates to
        config.rope_parameters. The cambrian-s code sets config.rope_scaling = None
        (originally harmless in transformers 4.x), which now clears rope_parameters
        and breaks Qwen2RotaryEmbedding initialization.
        """
        import torch.nn as nn
        from cambrian.model.language_model.cambrian_qwen2 import (
            CambrianQwenForCausalLM,
            CambrianQwenModel,
        )
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        def patched_init(self, config):
            rope_params = getattr(config, "rope_parameters", None)
            Qwen2ForCausalLM.__init__(self, config)
            config.model_type = "cambrian_qwen"
            if rope_params is not None:
                config.rope_parameters = rope_params
            self.model = CambrianQwenModel(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        CambrianQwenForCausalLM.__init__ = patched_init

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cambrian-S model instance."""
        from cambrian.model.builder import load_pretrained_model

        self._patch_cambrian_init()
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer, model, self.image_processor, _ = load_pretrained_model(
            pretrained_model_name,
            None,
            "cambrian-s",
            device="cpu",
            device_map={"": "cpu"},
        )

        if dtype_override is not None and dtype_override != torch.float16:
            model = model.to(dtype=dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for Cambrian-S."""
        from cambrian.mm_utils import process_images, tokenizer_image_token
        from cambrian.conversation import conv_templates
        from cambrian.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        image_tensor_list = process_images(
            [image], self.image_processor, self.model.config
        )
        images = image_tensor_list[0]

        if dtype_override is not None:
            images = images.to(dtype=dtype_override)

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(
            conv.roles[0],
            f"{DEFAULT_IMAGE_TOKEN}\nWhat is shown in this image?",
        )
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)

        attention_mask = torch.ones_like(input_ids)

        if batch_size > 1:
            input_ids = input_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)
            images = images.repeat(batch_size, 1, 1, 1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
        }

    def decode_output(self, outputs, input_length=None):
        """Decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self.load_model()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)
