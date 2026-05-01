# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral 3B Instruct BnB 4-bit model loader implementation for multimodal vision-language modeling.
"""

import torch
from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


def _patch_get_image_features():
    """Patch Mistral3Model.get_image_features to compute split_sizes on CPU.

    The split_sizes computation uses torch.as_tensor(image_sizes, device=TT_device)
    followed by prod(dim=-1). The TT prod reduction gives wrong results for integer
    products in the bfloat16 precision range (e.g. 42 * 55 → 2320 instead of 2310).
    Keep the integer metadata computation on CPU to avoid the device arithmetic bug.
    """
    import torch
    try:
        import transformers.models.mistral3.modeling_mistral3 as _m3
    except ImportError:
        return

    def _fixed_get_image_features(self, pixel_values, image_sizes, vision_feature_layer=None, output_hidden_states=None, **kwargs):
        # @merge_with_config_defaults is on the original method; replicate the None→config fallback here.
        if vision_feature_layer is None:
            vision_feature_layer = self.config.vision_feature_layer
        kwargs = {k: v for k, v in kwargs.items() if v is not None and k not in ("return_dict", "output_hidden_states")}
        image_outputs = self.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
        downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
        # Compute split_sizes on CPU to avoid TT device prod reduction bug.
        split_sizes = (image_sizes.cpu() // downsample_ratio).prod(dim=-1).tolist()
        image_features = torch.split(image_features.squeeze(0), split_sizes)
        image_outputs.pooler_output = image_features
        return image_outputs

    _m3.Mistral3Model.get_image_features = _fixed_get_image_features


def _replace_linear4bit(model):
    """Replace Linear4bit layers with regular nn.Linear layers.

    bitsandbytes 0.49+ auto-dequantizes on CPU when torch_dtype is set, so
    weights may already be plain Parameters. Handle both Params4bit and plain
    Parameter weights.
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        return model

    replacements = {}
    for name, module in model.named_modules():
        if not isinstance(module, Linear4bit):
            continue
        with torch.no_grad():
            w = module.weight
            if hasattr(w, "quant_state"):
                weight = bnb.functional.dequantize_4bit(w.data, w.quant_state)
            else:
                weight = w.data
            out_features, in_features = weight.shape
        has_bias = module.bias is not None
        new_layer = torch.nn.Linear(in_features, out_features, bias=has_bias)
        new_layer.weight = torch.nn.Parameter(weight)
        if has_bias:
            new_layer.bias = module.bias
        replacements[name] = new_layer

    for name, new_layer in replacements.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_layer)

    return model


class ModelVariant(StrEnum):
    """Available Ministral 3B Instruct BnB 4-bit model variants."""

    MINISTRAL_3B_INSTRUCT_2512_BNB_4BIT = (
        "unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit"
    )


class ModelLoader(ForgeModel):
    """Ministral 3B Instruct BnB 4-bit model loader for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3B_INSTRUCT_2512_BNB_4BIT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MINISTRAL_3B_INSTRUCT_2512_BNB_4BIT),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3B_INSTRUCT_2512_BNB_4BIT

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ministral_3b_instruct_bnb_4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        from transformers import AutoProcessor

        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ministral 3B Instruct BnB 4-bit model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance.
        """
        from transformers import Mistral3ForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # BnB variants need device_map="cpu" for CPU-based loading
        model_kwargs["device_map"] = "cpu"

        model_kwargs |= kwargs
        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model = _replace_linear4bit(model)
        _patch_get_image_features()

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        from PIL import Image
        from ....tools.utils import cast_input_to_type, get_file

        if self.processor is None:
            self._load_processor(dtype_override)

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt or self.sample_text},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = cast_input_to_type(
                    inputs["pixel_values"], dtype_override
                )

        return inputs
