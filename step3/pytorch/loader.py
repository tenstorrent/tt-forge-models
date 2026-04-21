# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Step3 VL model loader implementation for multimodal conditional generation.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from PIL import Image
from typing import Optional


def _compute_default_rope_parameters(config, device=None, **kwargs):
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

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
from ...tools.utils import get_file, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Step3 VL model variants."""

    STEP3_VL = "Step3_VL"


class ModelLoader(ForgeModel):
    """Step3 VL model loader implementation for multimodal conditional generation tasks."""

    _VARIANTS = {
        ModelVariant.STEP3_VL: ModelConfig(
            pretrained_model_name="stepfun-ai/step3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STEP3_VL

    sample_text = "What is shown in this image?"
    sample_image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Step3 VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if "pad_token_id" not in config.__dict__:
            config.pad_token_id = config.__dict__.get("eos_token_id", 0)
        if (
            hasattr(config, "text_config")
            and "pad_token_id" not in config.text_config.__dict__
        ):
            config.text_config.pad_token_id = config.pad_token_id

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.processor is None:
            self._load_processor()

        tokenizer = self.processor.tokenizer

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return tokenizer.decode(next_token_id)
