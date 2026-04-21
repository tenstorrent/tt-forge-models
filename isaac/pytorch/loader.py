# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isaac model loader implementation for multimodal visual question answering
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from PIL import Image
from typing import Optional
from ...tools.utils import get_file, cast_input_to_type
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

import transformers.cache_utils as _cache_utils
import transformers.image_processing_utils_fast as _img_fast
import transformers.tokenization_utils as _tok_utils

if not hasattr(_cache_utils, "SlidingWindowCache"):
    _cache_utils.SlidingWindowCache = type(
        "SlidingWindowCache", (_cache_utils.StaticCache,), {}
    )

if not hasattr(_img_fast, "DefaultFastImageProcessorKwargs"):
    _img_fast.DefaultFastImageProcessorKwargs = _img_fast.ImagesKwargs

if not hasattr(_tok_utils, "TensorType"):
    from transformers.utils import TensorType

    _tok_utils.TensorType = TensorType


def _fix_rope_parameters(config):
    """Ensure text_config.rope_parameters contains rope_theta."""
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        return
    rope_params = getattr(text_cfg, "rope_parameters", None)
    if rope_params is not None and "rope_theta" not in rope_params:
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(text_cfg, "rope_theta", 1000000.0)
        rope_params["rope_theta"] = rope_theta


class ModelVariant(StrEnum):
    """Available Isaac model variants."""

    ISAAC_0_2_2B_PREVIEW = "0.2_2B_Preview"


class ModelLoader(ForgeModel):
    """Isaac model loader implementation for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.ISAAC_0_2_2B_PREVIEW: ModelConfig(
            pretrained_model_name="PerceptronAI/Isaac-0.2-2B-Preview",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ISAAC_0_2_2B_PREVIEW

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Isaac",
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
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        _fix_rope_parameters(config)
        config._attn_implementation = "sdpa"
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config._attn_implementation = "sdpa"
        if hasattr(config, "vision_config") and config.vision_config is not None:
            config.vision_config._attn_implementation = "sdpa"

        model_kwargs = {
            "trust_remote_code": True,
            "config": config,
            "attn_implementation": "sdpa",
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
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        conversation = [
            {
                "role": "user",
                "content": "<image>\nWhat is shown in this image?",
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

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
            return self.processor.decode(next_token_id)
