# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HyperCLOVAX SEED Vision Instruct model loader implementation for multimodal visual question answering
"""

import torch
from typing import TypedDict
import transformers.processing_utils as _pu
import transformers.modeling_utils as _mu

if not hasattr(_pu, "ChatTemplateLoadKwargs"):
    _pu.ChatTemplateLoadKwargs = TypedDict("ChatTemplateLoadKwargs", {})
    _pu.AllKwargsForChatTemplate.__annotations__[
        "mm_load_kwargs"
    ] = _pu.ChatTemplateLoadKwargs

if not hasattr(_mu, "no_init_weights"):
    from transformers.initialization import no_init_weights

    _mu.no_init_weights = no_init_weights

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from typing import Optional
from ....tools.utils import get_file, cast_input_to_type
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


class ModelVariant(StrEnum):
    """Available HyperCLOVAX SEED Vision Instruct model variants."""

    SEED_VISION_INSTRUCT_3B = "SEED_Vision_Instruct_3B"


class ModelLoader(ForgeModel):
    """HyperCLOVAX SEED Vision Instruct model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.SEED_VISION_INSTRUCT_3B: ModelConfig(
            pretrained_model_name="naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEED_VISION_INSTRUCT_3B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HyperCLOVAX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True, "_attn_implementation": "eager"}
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model

        return model

    @staticmethod
    def _cast_nested(obj, dtype):
        if torch.is_tensor(obj):
            return cast_input_to_type(obj, dtype)
        if isinstance(obj, list):
            return [ModelLoader._cast_nested(item, dtype) for item in obj]
        return obj

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)
                elif isinstance(inputs[key], list):
                    inputs[key] = self._cast_nested(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        arguments = {
            **inputs,
            "use_cache": False,
            "max_new_tokens": 20,
            "do_sample": False,
        }

        return arguments

    def decode_output(self, outputs, input_length=None):
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)
