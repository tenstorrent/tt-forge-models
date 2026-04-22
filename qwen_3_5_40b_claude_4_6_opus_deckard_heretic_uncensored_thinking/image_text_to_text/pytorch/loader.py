# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DavidAU/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking model
loader for image-text-to-text generation.
"""

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
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
    """Available Qwen3.5-40B Claude 4.6 Opus Deckard Heretic model variants."""

    QWEN_3_5_40B_CLAUDE_4_6_OPUS_DECKARD_HERETIC = (
        "40b_claude_4_6_opus_deckard_heretic_uncensored_thinking"
    )


class ModelLoader(ForgeModel):
    """DavidAU Qwen3.5-40B Claude 4.6 Opus Deckard Heretic model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_40B_CLAUDE_4_6_OPUS_DECKARD_HERETIC: ModelConfig(
            pretrained_model_name="DavidAU/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_40B_CLAUDE_4_6_OPUS_DECKARD_HERETIC

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DavidAU Qwen3.5-40B Claude 4.6 Opus Deckard Heretic",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DavidAU Qwen3.5-40B Claude 4.6 Opus Deckard Heretic model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def _get_text_config(self):
        """Get the text config from the model config."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
        return config.text_config

    def get_mesh_config(self, num_devices: int):
        model_dim = 4  # constrained by num_key_value_heads=4
        batch_dim = max(1, num_devices // model_dim)
        mesh_shape = (batch_dim, model_dim)
        text_config = self._get_text_config()
        assert (
            text_config.num_attention_heads % model_dim == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Shard full-attention and MLP layers for tensor parallel."""
        shard_specs = {}
        lm = model.model.language_model
        for layer in lm.layers:
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DavidAU Qwen3.5-40B Claude 4.6 Opus Deckard Heretic model."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
