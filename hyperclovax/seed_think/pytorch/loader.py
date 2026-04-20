# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HyperCLOVAX SEED Think model loader implementation for multimodal visual question answering
"""

import torch
import transformers.modeling_utils as _modeling_utils
from transformers.initialization import no_init_weights as _no_init_weights
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_INIT_FUNCTIONS

if not hasattr(_modeling_utils, "no_init_weights"):
    _modeling_utils.no_init_weights = _no_init_weights

if "default" not in _ROPE_INIT_FUNCTIONS:

    def _compute_default_rope_parameters(
        config=None, device=None, seq_len=None, **kwargs
    ):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        return inv_freq, 1.0

    _ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPooling
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
    """Available HyperCLOVAX SEED Think model variants."""

    SEED_THINK_32B = "SEED_Think_32B"


class ModelLoader(ForgeModel):
    """HyperCLOVAX SEED Think model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.SEED_THINK_32B: ModelConfig(
            pretrained_model_name="naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEED_THINK_32B

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
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        vision_model = getattr(getattr(model, "model", None), "vision_model", None)
        if vision_model is not None:
            orig_forward = vision_model.forward

            def _unwrap_forward(*args, **kwargs):
                out = orig_forward(*args, **kwargs)
                if isinstance(out, BaseModelOutputWithPooling):
                    return out.pooler_output
                return out

            vision_model.forward = _unwrap_forward

        self.model = model

        return model

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

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

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
