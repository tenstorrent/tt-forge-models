# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternLM-XComposer2 model loader implementation for multimodal visual question answering.
"""

import torch
import transformers.modeling_utils as _modeling_utils
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ...tools.utils import get_file
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


class ModelVariant(StrEnum):
    """Available InternLM-XComposer2 model variants."""

    INTERNLM_XCOMPOSER2_7B = "7B"
    INTERNLM_XCOMPOSER2_VL_7B = "VL_7B"


class ModelLoader(ForgeModel):
    """InternLM-XComposer2 model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.INTERNLM_XCOMPOSER2_7B: ModelConfig(
            pretrained_model_name="internlm/internlm-xcomposer2-7b",
        ),
        ModelVariant.INTERNLM_XCOMPOSER2_VL_7B: ModelConfig(
            pretrained_model_name="internlm/internlm-xcomposer2-vl-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERNLM_XCOMPOSER2_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="InternLM-XComposer2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "max_length"):
            config.max_length = 8192

        model_kwargs = {
            "config": config,
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": False,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.x wraps model __init__ in a meta device context, but the
        # custom InternLM-XComposer2 code calls CLIPVisionModel.from_pretrained()
        # inside __init__, which is rejected within a meta device context.
        # Monkeypatching check_and_set_device_map to skip the meta device guard
        # for the duration of this load call works around the incompatibility.
        _orig_check = _modeling_utils.check_and_set_device_map

        def _permissive_check(device_map):
            if device_map is None:
                return None
            return _orig_check(device_map)

        _modeling_utils.check_and_set_device_map = _permissive_check
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            _modeling_utils.check_and_set_device_map = _orig_check
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        if self.model is None:
            raise RuntimeError("Model must be loaded before inputs via load_model()")

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")
        image = self.model.vis_processor(image)

        if dtype_override is not None:
            image = image.to(dtype_override)

        # Build query with image placeholder
        query = "<ImageHere>What is shown in this image?"

        # Tokenize the query
        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if image.dim() == 3:
            image = image.unsqueeze(0)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            image = image.repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": image,
            "use_cache": False,
        }

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.tokenizer is None:
            self._load_tokenizer()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)
