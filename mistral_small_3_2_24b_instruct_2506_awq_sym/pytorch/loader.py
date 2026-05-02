# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
jeffcookio Mistral Small 3.2 24B Instruct 2506 AWQ symmetric model loader implementation for multimodal vision-language modeling.
"""

import types
from typing import Optional

import torch

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


def _patch_get_image_features_cpu_split(model):
    """Patch get_image_features to compute split_sizes on CPU.

    On TT device, int64 arithmetic is promoted to bfloat16 internally.
    bfloat16(2310) = 2320, so prod() on an XLA/TT int64 tensor returns
    wrong results (e.g., 42*55=2310 rounds to 2320). Computing split_sizes
    on CPU avoids this precision loss.

    Also captures return_dict as an explicit named parameter to avoid
    conflicts when the @can_return_tuple-decorated caller passes it.
    """

    def get_image_features(self, pixel_values, image_sizes, vision_feature_layer=None, output_hidden_states=None, return_dict=None, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        if vision_feature_layer is None:
            vision_feature_layer = self.config.vision_feature_layer
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
        downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
        # Compute split_sizes on CPU to avoid TT int64→bfloat16 precision loss:
        # prod() on XLA int64 uses bfloat16 arithmetic; bfloat16(2310)=2320.
        image_sizes_cpu = (
            image_sizes.cpu().to(torch.int64)
            if isinstance(image_sizes, torch.Tensor)
            else torch.tensor(image_sizes, dtype=torch.int64)
        )
        split_sizes = (image_sizes_cpu // downsample_ratio).prod(dim=-1).tolist()
        image_features = torch.split(image_features.squeeze(0), split_sizes)
        image_outputs.pooler_output = image_features
        return image_outputs

    # Patch on model.model (Mistral3Model) since that's where get_image_features is called
    model.model.get_image_features = types.MethodType(get_image_features, model.model)


class ModelVariant(StrEnum):
    """Available Mistral Small 3.2 24B Instruct 2506 AWQ symmetric model variants."""

    MISTRAL_SMALL_3_2_24B_INSTRUCT_2506_AWQ_SYM = "24B_Instruct_2506_AWQ_Sym"


class ModelLoader(ForgeModel):
    """jeffcookio Mistral Small 3.2 24B Instruct 2506 AWQ symmetric model loader for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_3_2_24B_INSTRUCT_2506_AWQ_SYM: LLMModelConfig(
            pretrained_model_name="jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_3_2_24B_INSTRUCT_2506_AWQ_SYM

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
            model="Mistral Small 3.2 24B Instruct 2506 AWQ Sym",
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["device_map"] = "cpu"
        model_kwargs |= kwargs

        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        _patch_get_image_features_cpu_split(model)

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
            # Feed-forward (PixtralMLP)
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.down_proj.weight] = ("batch", "model")

            # Attention (PixtralAttention)
            shard_specs[layer.attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.o_proj.weight] = ("batch", "model")

        return shard_specs
