# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LaViDa-LLaDA model loader implementation for image-text-to-text tasks.
"""

import torch
from contextlib import contextmanager
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
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.x wraps model __init__ in torch.device("meta"), which breaks
        # lavida's vision tower that calls from_pretrained inside __init__. Patch
        # check_and_set_device_map to return "cpu" instead of raising in that case.
        # Must patch in modeling_utils (where it's imported by name), not in accelerate.
        import transformers.modeling_utils as _t_modeling
        from transformers.modeling_utils import (
            get_torch_context_manager_or_global_device,
        )

        _orig_check = _t_modeling.check_and_set_device_map

        def _patched_check(device_map):
            if (
                device_map is None
                and get_torch_context_manager_or_global_device() == torch.device("meta")
            ):
                return "cpu"
            return _orig_check(device_map)

        def _load_model():
            return AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        def _patch_siglip_config():
            # SigLipVisionConfig.from_pretrained calls _set_token_in_kwargs which was
            # removed in transformers 5.x. Patch it after the module is in sys.modules.
            import sys

            for _mod in sys.modules.values():
                _cls = getattr(_mod, "SigLipVisionConfig", None)
                if _cls is not None and not hasattr(_cls, "_set_token_in_kwargs"):
                    _cls._set_token_in_kwargs = classmethod(lambda cls, kwargs: None)

        _t_modeling.check_and_set_device_map = _patched_check
        try:
            try:
                model = _load_model()
            except AttributeError as e:
                if "_set_token_in_kwargs" not in str(e):
                    raise
                _patch_siglip_config()
                model = _load_model()
        finally:
            _t_modeling.check_and_set_device_map = _orig_check
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
