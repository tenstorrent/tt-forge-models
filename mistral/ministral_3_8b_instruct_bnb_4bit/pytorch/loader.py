# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral 3 8B Instruct BnB 4-bit model loader implementation for multimodal vision-language modeling.
"""

import torch
import torch.nn as nn
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


class ModelVariant(StrEnum):
    """Available Ministral 3 8B Instruct BnB 4-bit model variants."""

    MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT = (
        "unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit"
    )


class ModelLoader(ForgeModel):
    """Ministral 3 8B Instruct BnB 4-bit model loader for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT: LLMModelConfig(
            pretrained_model_name=str(
                ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT
            ),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT

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
            model="ministral_3_8b_instruct_bnb_4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _dequantize_bnb4_to_bf16(model):
        """Replace all BnB Linear4bit layers with standard bfloat16 Linear layers.

        Params4bit.detach() returns a plain Tensor, which causes
        Parameter.__new__ to raise RuntimeError when model.to(xla_device) is
        called. Dequantizing to bf16 before device transfer avoids this.
        """
        import bitsandbytes as bnb
        import bitsandbytes.functional as F

        replacements = []
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                if hasattr(module.weight, "quant_state") and module.weight.quant_state is not None:
                    weight_bf16 = F.dequantize_4bit(
                        module.weight.data, module.weight.quant_state
                    ).to(torch.bfloat16)
                else:
                    weight_bf16 = module.weight.data.to(torch.bfloat16)
                bias = module.bias
                new_linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=bias is not None,
                    device=weight_bf16.device,
                    dtype=torch.bfloat16,
                )
                new_linear.weight = nn.Parameter(weight_bf16)
                if bias is not None:
                    new_linear.bias = nn.Parameter(bias.to(torch.bfloat16))
                replacements.append((name, new_linear))

        for name, new_module in replacements:
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

        return model

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

    @staticmethod
    def _patch_get_image_features(model):
        """Patch Mistral3Model.get_image_features to compute split_sizes on CPU.

        The buggy line is in Mistral3Model (model.model), not the outer wrapper.
        TT device casts int64 image_sizes tensors to bf16 before arithmetic,
        corrupting values like 1540→1536 which makes split_sizes wrong (2320
        instead of 2310). Keeping the metadata computation on CPU avoids this.
        """
        import types

        def _get_image_features(self, pixel_values, image_sizes, vision_feature_layer=None, output_hidden_states=None, **kwargs):
            import torch
            # Resolve vision_feature_layer from config if not provided
            if vision_feature_layer is None:
                vision_feature_layer = self.config.vision_feature_layer
            # Strip return_dict and other output-control kwargs (handled by caller)
            kwargs = {k: v for k, v in kwargs.items() if v is not None and k not in ("return_dict",)}
            # Keep image_sizes on CPU throughout: TT device casts int64→bf16 (1540→1536).
            # Both vision_tower (PixtralVisionModel line 452) and multi_modal_projector
            # (Mistral3PatchMerger line 76) iterate over image_sizes in Python, which
            # triggers Error code: 13 (device-to-host) when the tensor is on TT device.
            image_sizes_cpu = torch.as_tensor(image_sizes, dtype=torch.long, device="cpu")
            image_outputs = self.vision_tower(
                pixel_values,
                image_sizes=image_sizes_cpu,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            if isinstance(vision_feature_layer, int):
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            else:
                hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
                selected_image_feature = torch.cat(hs_pool, dim=-1)

            image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes_cpu)
            downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
            split_sizes = (
                (image_sizes_cpu // downsample_ratio)
                .prod(dim=-1)
                .tolist()
            )
            image_features = torch.split(image_features.squeeze(0), split_sizes)
            image_outputs.pooler_output = image_features
            return image_outputs

        # Patch the inner Mistral3Model instance (model.model), not the outer wrapper
        model.model.get_image_features = types.MethodType(_get_image_features, model.model)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ministral 3 8B Instruct BnB 4-bit model instance.

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

        model.eval()

        self._dequantize_bnb4_to_bf16(model)
        self._patch_get_image_features(model)

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
