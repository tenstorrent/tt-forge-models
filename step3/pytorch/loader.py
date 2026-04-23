# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Step3 VL model loader implementation for multimodal conditional generation.
"""

import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import Optional

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

_LARGE_MODEL_SPACE_THRESHOLD = 10 * 1024 * 1024 * 1024  # 10 GB
_FALLBACK_CACHE_DIR = "/tmp/hf_cache_step3"


def _get_cache_dir():
    """Return a cache_dir with sufficient free space for large GGUF downloads.

    Falls back to /tmp when HF_HOME is on a nearly-full filesystem.
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub")
    try:
        os.makedirs(hub_dir, exist_ok=True)
        if shutil.disk_usage(hub_dir).free >= _LARGE_MODEL_SPACE_THRESHOLD:
            return None  # use the default HF cache
    except OSError:
        pass
    os.makedirs(_FALLBACK_CACHE_DIR, exist_ok=True)
    return _FALLBACK_CACHE_DIR


class ModelVariant(StrEnum):
    """Available Step3 VL model variants."""

    STEP3_VL = "Step3_VL"
    STEP3_VL_10B_GGUF_Q4_K_M = "Step3_VL_10B_GGUF_Q4_K_M"


class ModelLoader(ForgeModel):
    """Step3 VL model loader implementation for multimodal conditional generation tasks."""

    _VARIANTS = {
        ModelVariant.STEP3_VL: ModelConfig(
            pretrained_model_name="stepfun-ai/step3",
        ),
        ModelVariant.STEP3_VL_10B_GGUF_Q4_K_M: ModelConfig(
            pretrained_model_name="seanbailey518/Step3-VL-10B-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STEP3_VL

    _GGUF_FILES = {
        ModelVariant.STEP3_VL_10B_GGUF_Q4_K_M: "Step3-VL-10B-Q4_K_M.gguf",
    }

    _BASE_PROCESSOR_NAMES = {
        ModelVariant.STEP3_VL_10B_GGUF_Q4_K_M: "stepfun-ai/Step3-VL-10B",
    }

    sample_text = "What is shown in this image?"
    sample_image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @property
    def _gguf_file(self):
        return self._GGUF_FILES.get(self._variant)

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
        processor_name = self._BASE_PROCESSOR_NAMES.get(
            self._variant, self._variant_config.pretrained_model_name
        )
        cache_dir = _get_cache_dir()
        processor_kwargs = {"trust_remote_code": True}
        if cache_dir is not None:
            processor_kwargs["cache_dir"] = cache_dir
        self.processor = AutoProcessor.from_pretrained(processor_name, **processor_kwargs)

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        cache_dir = _get_cache_dir()
        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        if self._gguf_file is not None:
            model_kwargs["gguf_file"] = self._gguf_file
        if cache_dir is not None:
            model_kwargs["cache_dir"] = cache_dir
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
