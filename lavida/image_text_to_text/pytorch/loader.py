# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LaViDa-LLaDA model loader implementation for image-text-to-text tasks.
"""

import sys
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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

        # Load config with delay_load=True so the vision tower is not loaded during
        # __init__ (which runs inside transformers' meta device context in 5.x,
        # making nested from_pretrained calls fail).
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        config.delay_load = True
        # Clear mm_tunable_parts so the vision tower is not eagerly loaded during
        # __init__ via the mm_vision_tower branch in SigLipVisionTower.
        config.mm_tunable_parts = ""
        config.unfreeze_mm_vision_tower = False

        # Apply transformers 5.x compatibility patches to the custom model module.
        # The modeling_lavida.py module uses APIs removed in transformers 5.x.
        for mod_name, mod in sys.modules.items():
            if "modeling_lavida" not in mod_name:
                continue

            # Patch LLaDAModelLM.tie_weights to accept **kwargs:
            # transformers 5.x calls tie_weights(recompute_mapping=False).
            if hasattr(mod, "LLaDAModelLM"):
                cls = mod.LLaDAModelLM
                _orig_tie = cls.tie_weights
                if not getattr(_orig_tie, "_patched_kwargs", False):

                    def _tie_weights_compat(self, _orig=_orig_tie, **kwargs):
                        return _orig(self)

                    _tie_weights_compat._patched_kwargs = True
                    cls.tie_weights = _tie_weights_compat

            # Patch SigLipVisionConfig._set_token_in_kwargs:
            # removed in transformers 5.x; was a noop for the token-passing pattern.
            if hasattr(mod, "SigLipVisionConfig"):
                cfg_cls = mod.SigLipVisionConfig
                if not hasattr(cfg_cls, "_set_token_in_kwargs"):

                    @classmethod
                    def _set_token_in_kwargs(cls, kwargs, token=None):
                        pass

                    cfg_cls._set_token_in_kwargs = _set_token_in_kwargs

            break

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Manually load vision tower now that we're outside the meta device context.
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            vision_tower.load_model()
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
        img_proc = vision_tower.image_processor
        # SigLipImageProcessor exposes preprocess() rather than __call__.
        process_fn = img_proc if callable(img_proc) else img_proc.preprocess
        pixel_values = process_fn(images=image, return_tensors="pt")["pixel_values"]

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
