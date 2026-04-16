# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cambrian-S model loader implementation for multimodal visual question answering.
"""

import torch
from transformers import AutoModelForCausalLM
from typing import Optional

# Register CambrianQwen model type with transformers AutoModel/AutoConfig
from cambrian.model.language_model.cambrian_qwen2 import (
    CambrianQwenForCausalLM,
)  # noqa: F401


def _patch_cambrian_qwen_rope():
    """Fix compatibility between cambrian-s and newer transformers.

    CambrianQwenForCausalLM.__init__ sets config.rope_scaling = None after
    calling Qwen2ForCausalLM.__init__ but before creating CambrianQwenModel.
    In newer transformers, this invalidates the computed rope_parameters
    property, causing a TypeError in Qwen2RotaryEmbedding. This patch restores
    rope_parameters immediately after the rope_scaling reset.
    """
    from cambrian.model.language_model.cambrian_qwen2 import CambrianQwenModel
    from transformers import Qwen2ForCausalLM

    import torch.nn as nn

    def _patched_init(self, config):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "cambrian_qwen"
        rope_params = config.rope_parameters
        config.rope_scaling = None
        if config.rope_parameters is None and rope_params is not None:
            config.rope_parameters = rope_params
        self.model = CambrianQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    CambrianQwenForCausalLM.__init__ = _patched_init


_patch_cambrian_qwen_rope()

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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Cambrian-S model variants."""

    CAMBRIAN_S_7B_S3 = "S_7B_S3"


class ModelLoader(ForgeModel):
    """Cambrian-S model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.CAMBRIAN_S_7B_S3: ModelConfig(
            pretrained_model_name="nyu-visionx/Cambrian-S-7B-S3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CAMBRIAN_S_7B_S3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
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

    def _load_processor(self):
        from transformers import AutoTokenizer

        self.processor = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cambrian-S model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for Cambrian-S."""
        if self.processor is None:
            self._load_processor()

        tokenizer = self.processor
        IMAGE_TOKEN_INDEX = -200

        conversation = [
            {"role": "user", "content": "<image>\nWhat is shown in this image?"},
        ]
        text_prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Tokenize with IMAGE_TOKEN_INDEX replacing <image>
        chunks = text_prompt.split("<image>")
        input_ids = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                input_ids.append(IMAGE_TOKEN_INDEX)
            input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Create dummy image tensor for the SigLIP2 vision encoder (384x384)
        images = torch.randn(1, 3, 384, 384)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
        }

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, input_length=None):
        """Decode model outputs into human-readable text."""
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.processor.decode(next_token_id)
