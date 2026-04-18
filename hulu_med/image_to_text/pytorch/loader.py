# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hulu-Med model loader implementation for medical image-to-text generation.
"""

import functools
import inspect

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from typing import Optional

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
from ....tools.utils import get_file


class HuluMedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, attention_mask, inputs_embeds):
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=None,
            grid_sizes=None,
            merge_sizes=None,
            modals=None,
        )
        return outputs.logits


class ModelVariant(StrEnum):
    """Available Hulu-Med model variants for medical image-to-text."""

    HULU_MED_4B = "4B"


class ModelLoader(ForgeModel):
    """Hulu-Med model loader implementation for medical image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.HULU_MED_4B: ModelConfig(
            pretrained_model_name="ZJU-AI4H/Hulu-Med-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HULU_MED_4B

    sample_image = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._raw_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Hulu-Med",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_processor_compat(processor_cls):
        orig_get_args = processor_cls._get_arguments_from_pretrained
        sig = inspect.signature(orig_get_args)
        if "processor_dict" not in sig.parameters:

            @classmethod
            @functools.wraps(orig_get_args.__func__)
            def _patched_get_args(cls, pretrained_name, processor_dict=None, **kw):
                return orig_get_args.__func__(cls, pretrained_name, **kw)

            processor_cls._get_arguments_from_pretrained = _patched_get_args

        orig_merge = processor_cls._merge_kwargs

        def _patched_merge(
            self_proc, ModelProcessorKwargs, tokenizer_init_kwargs=None, **kwargs
        ):
            annotations = ModelProcessorKwargs.__annotations__
            output_kwargs = {
                "text_kwargs": {},
                "images_kwargs": {},
                "audio_kwargs": {},
                "videos_kwargs": {},
                "chat_template_kwargs": {},
                "common_kwargs": {},
            }
            default_kwargs = {k: {} for k in output_kwargs}
            used_keys = set()

            if tokenizer_init_kwargs is None:
                tokenizer_init_kwargs = {}

            for modality in default_kwargs:
                default_kwargs[modality] = ModelProcessorKwargs._defaults.get(
                    modality, {}
                ).copy()
                if modality in annotations and hasattr(
                    annotations[modality], "__annotations__"
                ):
                    for modality_key in annotations[modality].__annotations__:
                        if modality_key in tokenizer_init_kwargs:
                            value = (
                                getattr(self_proc.tokenizer, modality_key)
                                if hasattr(self_proc.tokenizer, modality_key)
                                else tokenizer_init_kwargs[modality_key]
                            )
                            default_kwargs[modality][modality_key] = value

            output_kwargs.update(default_kwargs)
            non_modality_kwargs = set(kwargs) - set(output_kwargs)

            for modality in output_kwargs:
                if modality in annotations and hasattr(
                    annotations[modality], "__annotations__"
                ):
                    for modality_key in annotations[modality].__annotations__:
                        if modality in kwargs:
                            kwarg_value = kwargs[modality].pop(
                                modality_key, "__empty__"
                            )
                            if (
                                kwarg_value != "__empty__"
                                and modality_key in non_modality_kwargs
                            ):
                                raise ValueError(
                                    f"Keyword argument {modality_key} was passed twice: "
                                    f"in a dictionary for {modality} and as a **kwarg."
                                )
                        elif modality_key in kwargs:
                            kwarg_value = kwargs.get(modality_key, "__empty__")
                        else:
                            kwarg_value = "__empty__"
                        if kwarg_value != "__empty__":
                            output_kwargs[modality][modality_key] = kwarg_value
                            used_keys.add(modality_key)

            if any(key in default_kwargs for key in kwargs):
                for modality, subdict in kwargs.items():
                    if modality in default_kwargs and isinstance(subdict, dict):
                        for subkey, subvalue in subdict.items():
                            if subkey not in used_keys:
                                output_kwargs[modality][subkey] = subvalue
                                used_keys.add(subkey)
            else:
                for key in kwargs:
                    if key not in used_keys:
                        output_kwargs["common_kwargs"][key] = kwargs[key]

            return output_kwargs

        processor_cls._merge_kwargs = _patched_merge

    def _load_processor(self):
        pretrained = self._variant_config.pretrained_model_name
        processor_cls = get_class_from_dynamic_module(
            "processing_hulumed.HulumedProcessor", pretrained
        )
        self._patch_processor_compat(processor_cls)
        self.processor = processor_cls.from_pretrained(
            pretrained, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hulu-Med model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Hulu-Med model instance for medical image-to-text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._raw_model = model

        return HuluMedWrapper(model)

    def _precompute_vision_inputs(self, inputs, dtype_override=None):
        model = self._raw_model
        with torch.no_grad():
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"].to(model.dtype),
                grid_sizes=inputs["grid_sizes"],
                merge_sizes=inputs["merge_sizes"],
                modals=inputs.get("modals"),
            )

        if dtype_override is not None and inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(dtype_override)

        return {
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
        }

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = str(get_file(self.sample_image))

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": {
                            "image_path": image_file,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this image.",
                    },
                ],
            }
        ]

        batch_inputs = self.processor(
            conversation=conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = dict(batch_inputs)

        result = self._precompute_vision_inputs(inputs, dtype_override)

        if batch_size > 1:
            for key in list(result.keys()):
                if torch.is_tensor(result[key]):
                    result[key] = result[key].repeat_interleave(batch_size, dim=0)

        return result

    def decode_output(self, outputs, dtype_override=None, inputs=None):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Model output tensors.
            dtype_override: Optional torch.dtype (unused, for API compatibility).
            inputs: Optional input tensors (unused, for API compatibility).

        Returns:
            str: Decoded text output.
        """
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.processor.decode(next_token_id)
