# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HyperCLOVAX SEED Omni model loader implementation for multimodal visual question answering
"""

import types

import torch
import torch.nn as nn
import transformers.modeling_utils
from transformers.initialization import no_init_weights
from transformers.modeling_outputs import BaseModelOutputWithPooling

transformers.modeling_utils.no_init_weights = no_init_weights

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


def _patch_vision_model_for_transformers5(model):
    """Fix transformers 5.x compat: Qwen2.5 VL vision model returns
    BaseModelOutputWithPooling instead of a plain tensor."""
    vision_model = model.model.vision_model

    # The HCX model code replaces the merger with nn.Identity and monkey-patches
    # the vision model forward. This breaks with transformers 5.x. Undo both and
    # wrap the forward to extract the merged tensor from the structured output.
    if isinstance(vision_model.merger, nn.Identity):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLPatchMerger,
        )

        vision_config = model.config.vision_config
        merger = Qwen2_5_VLPatchMerger(
            dim=vision_config.out_hidden_size, context_dim=vision_config.hidden_size
        )
        if hasattr(model, "dtype"):
            merger = merger.to(model.dtype)
        vision_model.merger = merger

    if "forward" in vision_model.__dict__:
        del vision_model.__dict__["forward"]

    orig_forward = vision_model.forward

    def _unwrap_forward(*args, **kwargs):
        output = orig_forward(*args, **kwargs)
        if isinstance(output, BaseModelOutputWithPooling):
            return output.pooler_output
        return output

    vision_model.forward = _unwrap_forward


class ModelVariant(StrEnum):
    """Available HyperCLOVAX SEED Omni model variants."""

    SEED_OMNI_8B = "SEED_Omni_8B"


class ModelLoader(ForgeModel):
    """HyperCLOVAX SEED Omni model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.SEED_OMNI_8B: ModelConfig(
            pretrained_model_name="naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEED_OMNI_8B

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
        _patch_vision_model_for_transformers5(model)
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
