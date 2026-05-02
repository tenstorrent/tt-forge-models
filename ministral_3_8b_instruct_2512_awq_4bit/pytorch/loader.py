# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
cyankiwi Ministral 3 8B Instruct 2512 AWQ 4-bit model loader implementation for multimodal vision-language modeling.
"""

from typing import Optional

from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Ministral 3 8B Instruct 2512 AWQ 4-bit model variants."""

    MINISTRAL_3_8B_INSTRUCT_2512_AWQ_4BIT = "3_8B_Instruct_2512_AWQ_4bit"


class ModelLoader(ForgeModel):
    """cyankiwi Ministral 3 8B Instruct 2512 AWQ 4-bit model loader for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_AWQ_4BIT: LLMModelConfig(
            pretrained_model_name="cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_AWQ_4BIT

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
            model="Ministral 3 8B Instruct 2512 AWQ 4-bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import AutoProcessor

        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    @staticmethod
    def _dequantize_compressed_tensors_layers(model, dtype):
        import torch.nn as nn
        from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor

        for parent_module in list(model.modules()):
            for child_name, child_module in list(parent_module.named_children()):
                if not (isinstance(child_module, nn.Linear) and hasattr(child_module, "quantization_scheme")):
                    continue
                state_dict = {
                    "weight_packed": child_module.weight_packed,
                    "weight_scale": child_module.weight_scale,
                    "weight_shape": child_module.weight_shape,
                }
                decompressed = PackedQuantizationCompressor.decompress(state_dict, child_module.quantization_scheme)
                weight_fp = decompressed["weight"].to(dtype).contiguous()
                bias = child_module.bias
                new_linear = nn.Linear(
                    child_module.in_features,
                    child_module.out_features,
                    bias=bias is not None,
                    dtype=dtype,
                )
                new_linear.weight = nn.Parameter(weight_fp)
                if bias is not None:
                    new_linear.bias = nn.Parameter(bias.to(dtype))
                setattr(parent_module, child_name, new_linear)

    @staticmethod
    def _patch_get_image_features(model):
        import types
        import torch

        inner = model.model

        def patched_get_image_features(
            self,
            pixel_values,
            image_sizes,
            vision_feature_layer=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
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
            # Compute split_sizes on CPU to avoid int64->bfloat16 rounding on TT device
            split_sizes = (
                (torch.as_tensor(image_sizes, dtype=torch.int64) // downsample_ratio).prod(dim=-1).tolist()
            )
            image_features = torch.split(image_features.squeeze(0), split_sizes)
            image_outputs.pooler_output = image_features
            return image_outputs

        inner.get_image_features = types.MethodType(patched_get_image_features, inner)

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        from transformers import Mistral3ForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        model_kwargs = {}
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["device_map"] = "cpu"
        model_kwargs |= kwargs

        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        self._dequantize_compressed_tensors_layers(model, dtype)
        self._patch_get_image_features(model)

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
        from PIL import Image
        from ...tools.utils import cast_input_to_type, get_file

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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}

        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        for layer in model.model.vision_tower.transformer.layers:
            shard_specs[layer.attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.o_proj.weight] = ("batch", "model")

            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.down_proj.weight] = ("batch", "model")

        return shard_specs
