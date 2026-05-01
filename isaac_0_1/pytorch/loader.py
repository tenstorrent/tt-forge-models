# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isaac 0.1 model loader implementation for multimodal visual question answering.
"""

import transformers.utils.generic as _trf_generic

if not hasattr(_trf_generic, "check_model_inputs"):
    _trf_generic.check_model_inputs = lambda fn: fn

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
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


def _patch_isaac_vision_embeddings(model):
    """Patch IsaacVisionEmbeddings.resize_positional_embeddings to cast to float32
    before F.interpolate with antialias=True, which is not supported for bfloat16/float16
    on any device (the original code only casts on CPU)."""
    for module in model.modules():
        if module.__class__.__name__ == "IsaacVisionEmbeddings":
            cls = module.__class__
            if getattr(cls, "_tt_resize_patched", False):
                return

            @staticmethod
            def resize_positional_embeddings(positional_embeddings, spatial_shapes, max_length):
                batch_size = spatial_shapes.shape[0]
                embed_dim = positional_embeddings.shape[-1]
                source_dtype = positional_embeddings.dtype

                resulted_positional_embeddings = torch.empty(
                    (batch_size, max_length, embed_dim),
                    device=positional_embeddings.device,
                    dtype=source_dtype,
                )

                positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

                # antialias=True not supported for bfloat16/float16 on any device
                if positional_embeddings.dtype in (torch.float16, torch.bfloat16):
                    positional_embeddings = positional_embeddings.to(torch.float32)

                for i in range(batch_size):
                    height, width = spatial_shapes[i]
                    resized_embeddings = F.interpolate(
                        positional_embeddings,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                    resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)
                    resized_embeddings = resized_embeddings.to(source_dtype)
                    resulted_positional_embeddings[i, : height * width] = resized_embeddings
                    resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

                return resulted_positional_embeddings

            cls.resize_positional_embeddings = resize_positional_embeddings
            cls._tt_resize_patched = True
            return


class ModelVariant(StrEnum):
    """Available Isaac 0.1 model variants."""

    ISAAC_0_1 = "Isaac_0_1"


class ModelLoader(ForgeModel):
    """Isaac 0.1 model loader implementation for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.ISAAC_0_1: ModelConfig(
            pretrained_model_name="PerceptronAI/Isaac-0.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ISAAC_0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Isaac-0.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        _patch_isaac_vision_embeddings(model)
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": "<image>What is shown in this image?",
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

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
