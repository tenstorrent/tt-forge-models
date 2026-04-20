# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MolmoAct model loader implementation for vision-language-action prediction.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available MolmoAct model variants."""

    MOLMOACT_7B_D_LIBERO_GOAL_0812 = "7B_D_LIBERO_Goal_0812"


class ModelLoader(ForgeModel):
    """MolmoAct model loader for vision-language-action prediction on robotic manipulation tasks."""

    _VARIANTS = {
        ModelVariant.MOLMOACT_7B_D_LIBERO_GOAL_0812: LLMModelConfig(
            pretrained_model_name="allenai/MolmoAct-7B-D-LIBERO-Goal-0812",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMOACT_7B_D_LIBERO_GOAL_0812

    sample_instruction = "put the wine bottle on top of the cabinet"
    sample_image_urls = [
        "https://huggingface.co/allenai/MolmoAct-7B-D-LIBERO-Goal/resolve/main/example_1.png",
        "https://huggingface.co/allenai/MolmoAct-7B-D-LIBERO-Goal/resolve/main/example_2.png",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MolmoAct",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
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

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        images = [
            Image.open(get_file(url)).convert("RGB") for url in self.sample_image_urls
        ]

        instruction = self.sample_instruction
        prompt = (
            f"The task is {instruction}. "
            "What is the action that the robot should take. "
            f"To figure out the action that the robot should take to {instruction}, "
            "let's think through it step by step. "
            "First, what is the depth map for the first image? "
            "Second, what is the trajectory of the end effector in the first image? "
            "Based on the depth map of the first image and the trajectory of the end effector "
            "in the first image, along with other images from different camera views as "
            "additional information, what is the action that the robot should take?"
        )

        text = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            images=[images],
            text=text,
            padding=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
