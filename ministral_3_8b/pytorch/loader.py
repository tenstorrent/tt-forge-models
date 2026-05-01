# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral 3-8B model loader implementation for multimodal vision-language modeling.
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
    """Available Ministral 3-8B model variants."""

    MINISTRAL_3_8B_INSTRUCT = "unsloth/Ministral-3-8B-Instruct-2512"


class ModelLoader(ForgeModel):
    """Ministral 3-8B model loader implementation for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MINISTRAL_3_8B_INSTRUCT),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_8B_INSTRUCT

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
            model="ministral_3_8b",
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
        """Load and return the Ministral 3-8B model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Ministral 3-8B model instance.
        """
        from transformers import Mistral3ForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        self._patch_get_image_features(model)

        model.eval()
        self.model = model
        self.config = model.config
        return model

    @staticmethod
    def _patch_get_image_features(model):
        """Patch get_image_features to compute split_sizes on CPU.

        TT silicon gives incorrect results for integer arithmetic (// and prod)
        on device tensors when image_sizes flows through the XLA graph.  The
        formula (image_sizes // downsample_ratio).prod(-1) returns 2320 instead
        of the correct 2310, causing torch.split to raise.  Computing on CPU
        avoids the TT integer-arithmetic bug.

        The instance-method replacement also bypasses the class-level
        @can_return_tuple and @merge_with_config_defaults decorators, so this
        function must replicate their relevant behaviours:
        - pop 'return_dict' from kwargs (can_return_tuple strips it for the body)
        - default vision_feature_layer from config when None (merge_with_config_defaults)
        """
        import types

        def _patched_get_image_features(self_inner, pixel_values, image_sizes, vision_feature_layer=None, **kwargs):
            import torch

            # @can_return_tuple pops return_dict; replicate that here.
            kwargs.pop("return_dict", None)
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            # @merge_with_config_defaults supplies vision_feature_layer from config.
            if vision_feature_layer is None:
                vision_feature_layer = self_inner.config.vision_feature_layer

            # Pass return_dict via kwargs so vision_tower's @can_return_tuple handles it.
            image_outputs = self_inner.vision_tower(
                pixel_values,
                image_sizes=image_sizes,
                output_hidden_states=True,
                **{**kwargs, "return_dict": True},
            )
            if isinstance(vision_feature_layer, int):
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            else:
                hs_pool = [image_outputs.hidden_states[i] for i in vision_feature_layer]
                selected_image_feature = torch.cat(hs_pool, dim=-1)

            image_features = self_inner.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
            downsample_ratio = self_inner.vision_tower.patch_size * self_inner.config.spatial_merge_size
            # Compute on CPU to avoid TT-silicon integer-arithmetic giving wrong results.
            split_sizes = [int(s) for s in (image_sizes.cpu() // downsample_ratio).prod(dim=-1).tolist()]
            image_features = torch.split(image_features.squeeze(0), split_sizes)
            image_outputs.pooler_output = image_features
            return image_outputs

        model.model.get_image_features = types.MethodType(_patched_get_image_features, model.model)

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the Ministral 3-8B model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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
        """Get the mesh configuration for tensor parallel execution."""
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution."""
        shard_specs = {}

        # Mistral3ForConditionalGeneration wraps Mistral3Model in model.model;
        # language_model and vision_tower are children of model.model.
        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        # vision_tower is a PixtralVisionModel: layers are in .transformer.layers;
        # attention is at layer.attention and MLP is at layer.feed_forward.
        for layer in model.model.vision_tower.transformer.layers:
            shard_specs[layer.attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.o_proj.weight] = ("batch", "model")

            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.down_proj.weight] = ("batch", "model")

        return shard_specs
