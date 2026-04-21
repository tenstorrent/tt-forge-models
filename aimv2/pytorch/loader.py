# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AIMv2 model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import torch

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import (
    VisionPreprocessor,
    VisionPostprocessor,
)
from datasets import load_dataset


# Revision pin recommended in the apple/aimv2-large-patch14-224-lit model card.
_LIT_REVISION = "c2cd59a786c4c06f39d199c50d08cc2eab9f8605"


@dataclass
class AIMv2Config(ModelConfig):
    """Configuration specific to AIMv2 models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available AIMv2 model variants."""

    LARGE_PATCH14_224_APPLE_PT_DIST = "Large_Patch14_224_Apple_PT_Dist"
    LARGE_PATCH14_224_LIT = "Large_Patch14_224_LIT"


class ModelLoader(ForgeModel):
    """AIMv2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_PATCH14_224_APPLE_PT_DIST: AIMv2Config(
            pretrained_model_name="hf_hub:timm/aimv2_large_patch14_224.apple_pt_dist",
            source=ModelSource.TIMM,
        ),
        ModelVariant.LARGE_PATCH14_224_LIT: AIMv2Config(
            pretrained_model_name="apple/aimv2-large-patch14-224-lit",
            source=ModelSource.HUGGING_FACE,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_PATCH14_224_APPLE_PT_DIST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None
        self._postprocessor = None
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        if source == ModelSource.HUGGING_FACE:
            task = ModelTask.CV_ZS_IMAGE_CLS
        else:
            task = ModelTask.CV_IMAGE_CLS

        return ModelInfo(
            model="AIMv2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=task,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            from transformers import AutoModel

            model_kwargs = {
                "trust_remote_code": True,
                "revision": _LIT_REVISION,
                "return_dict": False,
            }
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs.update(kwargs)

            model = AutoModel.from_pretrained(model_name, **model_kwargs)
            model.eval()

            self.model = model
            return model

        import timm

        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        source = self._variant_config.source

        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        if source == ModelSource.HUGGING_FACE:
            from transformers import AutoProcessor

            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(
                    self._variant_config.pretrained_model_name,
                    revision=_LIT_REVISION,
                )

            self.text_prompts = [
                "Picture of a dog.",
                "Picture of a cat.",
                "Picture of a horse.",
            ]

            inputs = self.processor(
                images=image,
                text=self.text_prompts,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

            if dtype_override is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

            return inputs

        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        model_for_config = None
        if hasattr(self, "model") and self.model is not None:
            model_for_config = self.model

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )

    def output_postprocess(self, output):
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            logits_per_image = output[0] if isinstance(output, tuple) else output
            probs = logits_per_image.softmax(dim=-1)
            prompts = self.text_prompts or []
            for i, text in enumerate(prompts):
                print(f"Probability of '{text}':", probs[0, i].item())
            return probs

        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
