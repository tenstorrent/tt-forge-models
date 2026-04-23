# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LaViDa-LLaDA model loader implementation for image-text-to-text tasks.
"""

import contextlib
import inspect
import sys
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
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
from ....tools.utils import get_file

IMAGE_TOKEN_INDEX = -200


def _patch_custom_model_classes():
    # SigLipVisionConfig.from_pretrained calls cls._set_token_in_kwargs, removed in
    # transformers 5.x. Restore as a no-op on the base class.
    if not hasattr(PretrainedConfig, "_set_token_in_kwargs"):

        @classmethod  # type: ignore[misc]
        def _set_token_in_kwargs(cls, kwargs, token=None):
            pass

        PretrainedConfig._set_token_in_kwargs = _set_token_in_kwargs

    # LLaDAModelLM.tie_weights() has no **kwargs but transformers 5.x calls
    # tie_weights(recompute_mapping=False). Patch any loaded class in this module
    # that overrides tie_weights without accepting keyword arguments.
    for module in sys.modules.values():
        if not hasattr(module, "__file__") or module.__file__ is None:
            continue
        if "modeling_lavida" not in module.__file__:
            continue
        for attr in vars(module).values():
            if not (isinstance(attr, type) and hasattr(attr, "tie_weights")):
                continue
            method = vars(attr).get("tie_weights")
            if method is None:
                continue
            sig = inspect.signature(method)
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not has_var_keyword:
                orig = method

                def _make_patched(fn):
                    def patched(self, **kwargs):
                        return fn(self)

                    return patched

                attr.tie_weights = _make_patched(orig)


@contextlib.contextmanager
def _allow_nested_from_pretrained_in_meta_context():
    # Transformers 5.x always initializes models inside torch.device("meta") context.
    # Custom models that call from_pretrained inside __init__ (e.g. SigLipVisionTower)
    # hit a safety check that raises RuntimeError. We patch check_and_set_device_map to
    # return None instead of raising, letting the nested load proceed on CPU.
    import transformers.modeling_utils as _tm
    from transformers.modeling_utils import get_torch_context_manager_or_global_device

    original = _tm.check_and_set_device_map

    def _patched(device_map):
        if device_map is None:
            if get_torch_context_manager_or_global_device() == torch.device("meta"):
                return None
        return original(device_map)

    _tm.check_and_set_device_map = _patched
    try:
        yield
    finally:
        _tm.check_and_set_device_map = original


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

        # Pre-load config to trigger custom module imports into sys.modules, then
        # patch classes for transformers 5.x API incompatibilities.
        AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        _patch_custom_model_classes()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        with _allow_nested_from_pretrained_in_meta_context():
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
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
        # The model's forward always requires labels; image token positions use -100.
        labels = input_ids.clone()
        labels[labels == IMAGE_TOKEN_INDEX] = -100

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        vision_tower = self.model.get_vision_tower()
        processor = vision_tower.image_processor
        call_fn = processor if callable(processor) else processor.preprocess
        pixel_values = call_fn(images=image, return_tensors="pt")["pixel_values"]

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
            "labels": labels,
        }
