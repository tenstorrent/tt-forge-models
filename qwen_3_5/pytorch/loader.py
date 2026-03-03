# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 model loader implementation for multimodal modeling.
"""

from typing import Optional, Any

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Qwen 3.5 multimodal model variants."""

    QWEN_3_5_27B = "Qwen/Qwen3.5-27B"


class ModelLoader(ForgeModel):
    """Qwen 3.5 model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_27B),
        ),

    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B

    sample_text = "What animal is on the candy?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        if "35B" in variant.value:
            group = ModelGroup.RED
        else:
            group = ModelGroup.RED

        return ModelInfo(
            model="qwen_3_5_multimodal",
            variant=variant,
            group=group,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.5 multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 3.5 model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
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
        """Load and return sample inputs for the Qwen 3.5 multimodal model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        image_url = image_url or self.sample_image_url
        text_prompt = prompt or self.sample_text

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution.

        Args:
            num_devices: Number of devices to shard across.

        Returns:
            tuple: (mesh_shape, mesh_axis_names) where mesh_shape is (batch_dim, model_dim)
                   and mesh_axis_names are ("batch", "model").
        """
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution.

        Qwen 3.5 uses a hybrid architecture mixing Gated DeltaNet (linear_attn)
        and standard Attention (self_attn) layers in the language model, and a
        vision model with combined QKV projections.

        Args:
            model: The AutoModelForImageTextToText model instance.

        Returns:
            dict or None: Sharding specification, or None if not implemented for this variant.
        """
        shard_specs = {}

        for layer in model.model.visual.blocks:
            shard_specs[layer.attn.qkv.weight] = ("model", "batch")
            shard_specs[layer.attn.proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.linear_fc1.weight] = ("model", "batch")
            shard_specs[layer.mlp.linear_fc2.weight] = ("batch", "model")

        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            elif hasattr(layer, "linear_attn"):
                shard_specs[layer.linear_attn.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_z.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")

        shard_specs[model.lm_head.weight] = ("batch", "model")
        return shard_specs
