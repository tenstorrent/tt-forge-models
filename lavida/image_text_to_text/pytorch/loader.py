# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LaViDa-LLaDA model loader implementation for image-text-to-text tasks.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
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
from ....tools.utils import get_file

IMAGE_TOKEN_INDEX = -200


class ModelVariant(StrEnum):
    """Available LaViDa-LLaDA model variants for image-text-to-text tasks."""

    LAVIDA_LLADA_V1_0_INSTRUCT = "lavida_llada_v1_0_instruct"


class ModelLoader(ForgeModel):
    """LaViDa-LLaDA model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.LAVIDA_LLADA_V1_0_INSTRUCT: LLMModelConfig(
            pretrained_model_name="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LAVIDA_LLADA_V1_0_INSTRUCT

    sample_image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
    sample_text = "Describe this image in detail."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LaViDa-LLaDA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LaViDa-LLaDA model instance."""
        from transformers import PreTrainedModel, PretrainedConfig

        pretrained_model_name = self._variant_config.pretrained_model_name

        # _set_token_in_kwargs was removed in transformers 5.x (use_auth_token migration
        # complete), but the model's custom SigLipVisionConfig.from_pretrained still calls it.
        if not hasattr(PretrainedConfig, "_set_token_in_kwargs"):

            @classmethod
            def _set_token_in_kwargs(cls, kwargs, token=None):
                if token is not None:
                    kwargs["token"] = token

            PretrainedConfig._set_token_in_kwargs = _set_token_in_kwargs

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.x always wraps model.__init__ in torch.device("meta") context
        # (PreTrainedModel.get_init_context). This model's __init__ calls
        # SigLipVisionModel.from_pretrained for the vision tower, which transformers 5.x
        # rejects inside that context. Patch out the meta device to allow sub-model loading.
        _original = PreTrainedModel.get_init_context.__func__

        @classmethod
        def _no_meta_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            return [
                c
                for c in _original(cls, dtype, is_quantized, _is_ds_init_called)
                if not (isinstance(c, torch.device) and c.type == "meta")
            ]

        PreTrainedModel.get_init_context = _no_meta_init_context
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            PreTrainedModel.get_init_context = classmethod(_original)

        model.resize_token_embeddings(len(self.tokenizer))
        model.tie_weights()
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the LaViDa-LLaDA model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        messages = [
            {
                "role": "user",
                "content": f"<image>\n{self.sample_text}",
            }
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        pre, post = rendered.split("<image>", 1)
        pre_ids = self.tokenizer(
            pre, return_tensors="pt", add_special_tokens=False
        ).input_ids
        post_ids = self.tokenizer(
            post, return_tensors="pt", add_special_tokens=False
        ).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        vision_tower = self.model.get_vision_tower()
        pixel_values = vision_tower.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": pixel_values,
        }
