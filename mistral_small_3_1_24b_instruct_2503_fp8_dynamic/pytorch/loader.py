# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RedHatAI Mistral Small 3.1 24B FP8 dynamic quantized model loader implementation for multimodal vision-language modeling.
"""

import types
from typing import Optional

import torch
import torch.nn as nn

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


def _get_fp8_dtypes():
    fp8_dtypes = set()
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        if hasattr(torch, name):
            fp8_dtypes.add(getattr(torch, name))
    return fp8_dtypes


def _dequantize_fp8_to_bf16(model, target_dtype=torch.bfloat16):
    """Replace FP8-typed Linear weights with dequantized BF16 weights.

    TT hardware does not support FP8 computation. compressed-tensors stores
    model weights in float8_e4m3fn. This walks all nn.Linear modules, detects
    FP8-typed weights, dequantizes them using the stored weight_scale, and
    replaces the weight parameter with a BF16 tensor. Also sets
    quantization_enabled=False to suppress per-token activation quantization
    in the patched quantized_forward, preventing FP8 activations on TT.
    """
    fp8_dtypes = _get_fp8_dtypes()
    if not fp8_dtypes:
        return

    try:
        from compressed_tensors.quantization.lifecycle.forward import (
            dequantize as ct_dequantize,
        )
    except ImportError:
        ct_dequantize = None

    for _, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype not in fp8_dtypes:
            continue

        scale = getattr(module, "weight_scale", None)
        zero_point = getattr(module, "weight_zero_point", None)
        g_idx = getattr(module, "weight_g_idx", None)

        if ct_dequantize is not None and scale is not None:
            dequant_w = ct_dequantize(
                weight.data, scale=scale, zero_point=zero_point, g_idx=g_idx
            )
        else:
            dequant_w = weight.data.to(target_dtype)

        module.weight = nn.Parameter(dequant_w.to(target_dtype))
        # Disable the compressed-tensors quantized_forward to prevent
        # FP8 activation quantization on TT device
        module.quantization_enabled = False


def _patch_mistral3_split_sizes(model):
    """Patch Mistral3Model.get_image_features to compute split_sizes on CPU.

    The transformers implementation does:
        split_sizes = (torch.as_tensor(image_sizes, device=image_features.device)
                       // downsample_ratio).prod(dim=-1).tolist()
    On TT device, image_features.device is TT, so this creates a TT tensor and
    calls .tolist() on it — causing INTERNAL Error code 13 (pjrt-device-to-host-
    transfer, Tier B). Keeping the computation on CPU avoids the transfer.
    """
    inner_model = model.model  # Mistral3Model
    original_fn = type(inner_model).get_image_features

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

        if vision_feature_layer is None:
            vision_feature_layer = getattr(self.config, "vision_feature_layer", -1)

        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [
                image_outputs.hidden_states[idx] for idx in vision_feature_layer
            ]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(
            selected_image_feature.squeeze(0), image_sizes
        )
        downsample_ratio = (
            self.vision_tower.patch_size * self.config.spatial_merge_size
        )
        # Compute split_sizes on CPU to avoid bfloat16 int64 rounding and
        # pjrt-device-to-host-transfer when calling .tolist() on a TT tensor
        split_sizes = (
            (image_sizes.cpu().to(torch.int64) // downsample_ratio)
            .prod(dim=-1)
            .tolist()
        )
        image_features = torch.split(image_features.squeeze(0), split_sizes)
        image_outputs.pooler_output = image_features

        return image_outputs

    inner_model.get_image_features = types.MethodType(
        patched_get_image_features, inner_model
    )


class ModelVariant(StrEnum):
    """Available Mistral Small 3.1 24B FP8 dynamic model variants."""

    MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_FP8_DYNAMIC = "24B_Instruct_2503_FP8_Dynamic"


class ModelLoader(ForgeModel):
    """RedHatAI Mistral Small 3.1 24B FP8 dynamic model loader for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_FP8_DYNAMIC: LLMModelConfig(
            pretrained_model_name="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_FP8_DYNAMIC

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
            model="Mistral Small 3.1 24B FP8 Dynamic",
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

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import Mistral3ForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {"device_map": "cpu"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        _dequantize_fp8_to_bf16(model, target_dtype)
        _patch_mistral3_split_sizes(model)

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
