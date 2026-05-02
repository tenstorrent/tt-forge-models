# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Small 3.1 model loader implementation for multimodal vision-language modeling.
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
    """Available Mistral Small 3.1 model variants."""

    MISTRAL_SMALL_3_1_24B_INSTRUCT = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    MISTRAL_SMALL_3_1_24B_INSTRUCT_INT4_AWQ = (
        "OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym"
    )
    MISTRAL_SMALL_3_1_24B_INSTRUCT_UNSLOTH_BNB_4BIT = (
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit"
    )


def _dequantize_bnb4_to_bf16(model):
    """Replace all BnB Linear4bit layers with plain bf16 nn.Linear.

    Unsloth models may store weights in bfloat16 in the safetensors files even
    though the module structure uses Linear4bit (from quantization_config).  In
    that case module.weight is a plain Parameter (no quant_state) and we just
    cast it to bfloat16 directly without calling dequantize_4bit.
    """
    import bitsandbytes as bnb

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            replacements.append((name, module))
    for name, module in replacements:
        quant_state = getattr(module.weight, "quant_state", None)
        if quant_state is not None:
            dq_weight = bnb.functional.dequantize_4bit(
                module.weight.data, quant_state
            ).to(torch.bfloat16)
        else:
            dq_weight = module.weight.data.to(torch.bfloat16)
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


class ModelLoader(ForgeModel):
    """Mistral Small 3.1 model loader implementation for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT),
        ),
        ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_INT4_AWQ: LLMModelConfig(
            pretrained_model_name=str(
                ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_INT4_AWQ
            ),
        ),
        ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_UNSLOTH_BNB_4BIT: LLMModelConfig(
            pretrained_model_name=str(
                ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_UNSLOTH_BNB_4BIT
            ),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT

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
            model="mistral_small_3_1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _is_bnb_4bit(self) -> bool:
        return self._variant == ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_UNSLOTH_BNB_4BIT

    @property
    def _is_quantized(self) -> bool:
        return self._variant in (
            ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_INT4_AWQ,
            ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_UNSLOTH_BNB_4BIT,
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
        """Load and return the Mistral Small 3.1 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Mistral Small 3.1 model instance.
        """
        from transformers import Mistral3ForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        if self._is_quantized:
            model_kwargs["device_map"] = "cpu"
        model_kwargs |= kwargs
        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        if self._is_bnb_4bit:
            model = _dequantize_bnb4_to_bf16(model)

        # Patch get_image_features to compute split_sizes on CPU; TT promotes
        # int64 → bfloat16, which rounds 2310 → 2320 and breaks torch.split.
        def _patched_get_image_features(
            self_inner,
            pixel_values,
            image_sizes,
            vision_feature_layer=None,
            output_hidden_states=None,
            return_dict=None,
            **kw,
        ):
            kw = {k: v for k, v in kw.items() if v is not None}
            if vision_feature_layer is None:
                vision_feature_layer = self_inner.config.vision_feature_layer
            image_outputs = self_inner.vision_tower(
                pixel_values,
                image_sizes=image_sizes,
                output_hidden_states=True,
                return_dict=True,
                **kw,
            )
            if isinstance(vision_feature_layer, int):
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            else:
                hs_pool = [image_outputs.hidden_states[i] for i in vision_feature_layer]
                selected_image_feature = torch.cat(hs_pool, dim=-1)

            image_features = self_inner.multi_modal_projector(
                selected_image_feature.squeeze(0), image_sizes
            )
            downsample_ratio = (
                self_inner.vision_tower.patch_size * self_inner.config.spatial_merge_size
            )
            split_sizes = (
                (torch.as_tensor(image_sizes).cpu() // downsample_ratio)
                .prod(dim=-1)
                .tolist()
            )
            image_features = torch.split(image_features.squeeze(0), split_sizes)
            image_outputs.pooler_output = image_features
            return image_outputs

        model.model.get_image_features = types.MethodType(
            _patched_get_image_features, model.model
        )

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
        """Load and return sample inputs for the Mistral Small 3.1 model.

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

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution."""
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution."""
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
