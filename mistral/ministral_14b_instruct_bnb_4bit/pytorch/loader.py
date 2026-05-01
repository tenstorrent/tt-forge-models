# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral 14B Instruct BnB 4-bit model loader implementation for multimodal vision-language modeling.
"""

import types
from typing import Optional

import torch
import torch.nn as nn

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
    """Available Ministral 14B Instruct BnB 4-bit model variants."""

    MINISTRAL_14B_INSTRUCT_2512_BNB_4BIT = (
        "unsloth/Ministral-3-14B-Instruct-2512-unsloth-bnb-4bit"
    )


def _dequantize_bnb4_to_bf16(model):
    import bitsandbytes as bnb

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            replacements.append((name, module))
    for name, module in replacements:
        dq_weight = bnb.functional.dequantize_4bit(
            module.weight.data, module.weight.quant_state
        ).to(torch.bfloat16)
        new_linear = nn.Linear(
            dq_weight.shape[1],
            dq_weight.shape[0],
            bias=module.bias is not None,
            dtype=torch.bfloat16,
        )
        new_linear.weight = nn.Parameter(dq_weight)
        if module.bias is not None:
            new_linear.bias = nn.Parameter(module.bias.data.to(torch.bfloat16))
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_linear)
    return model


def _patch_mistral3_for_tt_device(model):
    """Patch Mistral3Model to fix TT-device incompatibilities."""
    inner = model.model

    # Fix 1: split_sizes computed with device=image_features.device → bfloat16 int64
    # rounding (2310 → 2320). Compute on CPU instead.
    original_get_image_features = inner.get_image_features

    def _patched_get_image_features(
        self_inner,
        pixel_values,
        image_sizes,
        vision_feature_layer=None,
        output_hidden_states=None,
        **kwargs,
    ):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self_inner.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [
                image_outputs.hidden_states[layer_idx]
                for layer_idx in vision_feature_layer
            ]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self_inner.multi_modal_projector(
            selected_image_feature.squeeze(0), image_sizes
        )
        downsample_ratio = (
            self_inner.vision_tower.patch_size * self_inner.config.spatial_merge_size
        )
        # Use CPU int64 to avoid bfloat16 rounding of prod() on TT device.
        split_sizes = (
            (
                torch.as_tensor(image_sizes, device="cpu").to(torch.int64)
                // downsample_ratio
            )
            .prod(dim=-1)
            .tolist()
        )
        image_features = torch.split(image_features.squeeze(0), split_sizes)
        image_outputs.pooler_output = image_features
        return image_outputs

    inner.get_image_features = types.MethodType(_patched_get_image_features, inner)

    # Fix 2: generate_block_attention_mask uses in-place assignment on TT tensor.
    # Replace the global function with a CPU-built version.
    try:
        import transformers.models.pixtral.modeling_pixtral as _pixtral_mod

        def _generate_block_attention_mask_cpu(patch_embeds_list, tensor):
            dtype = tensor.dtype
            seq_len = tensor.shape[1]
            d_min = torch.finfo(dtype).min
            # Build entirely on CPU, then move to tensor.device.
            causal_mask = torch.full(
                (seq_len, seq_len), fill_value=d_min, dtype=dtype, device="cpu"
            )
            block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
            block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
            for start, end in zip(block_start_idx, block_end_idx):
                causal_mask[start:end, start:end] = 0
            causal_mask = causal_mask[None, None, :, :].expand(
                tensor.shape[0], 1, -1, -1
            )
            return causal_mask.to(tensor.device)

        _pixtral_mod.generate_block_attention_mask = (
            _generate_block_attention_mask_cpu
        )
    except Exception:
        pass

    # Fix 3: masked_scatter cumsum OOM. Replace with token-level gather.
    # masked_scatter on seq_len×hidden_size (11.8M elements) → cumsum → 45 GB OOM.
    original_forward = inner.forward

    def _patched_forward(
        self_inner,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        image_sizes=None,
        **kwargs,
    ):
        if pixel_values is not None and input_ids is not None:
            # Resolve vision_feature_layer default.
            if vision_feature_layer is None:
                vision_feature_layer = self_inner.config.vision_feature_layer

            image_outputs = self_inner.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
            )
            # image_outputs.pooler_output is a tuple of per-image feature tensors.
            image_features_list = image_outputs.pooler_output
            image_features_cat = torch.cat(image_features_list, dim=0)

            # Token-level gather: replace image-placeholder tokens with features.
            inputs_embeds = self_inner.get_input_embeddings()(input_ids)
            image_token_id = self_inner.config.image_token_id
            token_mask = (input_ids == image_token_id).squeeze(0)  # [seq]
            num_img_tokens = int(token_mask.sum().item())

            if num_img_tokens > 0 and num_img_tokens == image_features_cat.shape[0]:
                img_indices = token_mask.nonzero(as_tuple=True)[0]
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[0, img_indices] = image_features_cat.to(
                    inputs_embeds.dtype
                )

            input_ids = None  # consumed

        return original_forward(
            input_ids=input_ids,
            pixel_values=None,  # already consumed above
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer if pixel_values is None else None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            image_sizes=None,  # consumed
            **kwargs,
        )

    inner.forward = types.MethodType(_patched_forward, inner)

    return model


class ModelLoader(ForgeModel):
    """Ministral 14B Instruct BnB 4-bit model loader for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_14B_INSTRUCT_2512_BNB_4BIT: LLMModelConfig(
            pretrained_model_name=str(
                ModelVariant.MINISTRAL_14B_INSTRUCT_2512_BNB_4BIT
            ),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_14B_INSTRUCT_2512_BNB_4BIT

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
            model="ministral_14b_instruct_bnb_4bit",
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
        """Load and return the Ministral 14B Instruct BnB 4-bit model instance.

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

        # Dequantize BnB 4-bit Linear4bit → nn.Linear (bfloat16) for TT device.
        model = _dequantize_bnb4_to_bf16(model)

        # Patch model methods to fix TT-device incompatibilities.
        model = _patch_mistral3_for_tt_device(model)

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
