# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bartowski xlangai Jedi-3B-1080p GGUF model loader implementation for image to text.
"""

import types

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Bartowski xlangai Jedi-3B-1080p GGUF variants for image to text."""

    XLANGAI_JEDI_3B_1080P_GGUF = "xlangai_jedi_3b_1080p_gguf"


class ModelLoader(ForgeModel):
    """Bartowski xlangai Jedi-3B-1080p GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.XLANGAI_JEDI_3B_1080P_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/xlangai_Jedi-3B-1080p-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLANGAI_JEDI_3B_1080P_GGUF

    GGUF_FILE = "xlangai_Jedi-3B-1080p-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Bartowski xlangai Jedi-3B-1080p GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # Patch get_placeholder_mask to avoid boolean-indexed tensor size computation,
        # which evaluates to 0 under XLA tracing and causes a false mismatch error.
        def _patched_get_placeholder_mask(
            self, input_ids, inputs_embeds, image_features=None, video_features=None
        ):
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(
                        self.config.image_token_id,
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )
                special_image_mask = special_image_mask.all(-1)
                special_video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(
                        self.config.video_token_id,
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )
                special_video_mask = special_video_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id
                special_video_mask = input_ids == self.config.video_token_id

            n_image_tokens = special_image_mask.sum()
            special_image_mask = (
                special_image_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            if image_features is not None:
                hidden_size = inputs_embeds.shape[-1]
                if int(n_image_tokens) * hidden_size != image_features.numel():
                    raise ValueError(
                        f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}"
                    )

            n_video_tokens = special_video_mask.sum()
            special_video_mask = (
                special_video_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            if video_features is not None:
                hidden_size = inputs_embeds.shape[-1]
                if int(n_video_tokens) * hidden_size != video_features.numel():
                    raise ValueError(
                        f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}"
                    )
            return special_image_mask, special_video_mask

        model.model.get_placeholder_mask = types.MethodType(
            _patched_get_placeholder_mask, model.model
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
