# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.2 Vision model loader implementation for multimodal modeling.
"""

from typing import Optional, Any

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

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
from ....tools.utils import cast_input_to_type, get_file
from PIL import Image


class ModelVariant(StrEnum):
    """Available Llama 3.2 Vision model variants."""

    LLAMA_3_2_90B_VISION_INST = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    LLAMA_3_2_90B_VISION = "meta-llama/Llama-3.2-90B-Vision"


class ModelLoader(ForgeModel):
    """Gemma3 model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_90B_VISION_INST: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.LLAMA_3_2_90B_VISION_INST),
        ),
        ModelVariant.LLAMA_3_2_90B_VISION: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.LLAMA_3_2_90B_VISION),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_90B_VISION_INST

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        group = ModelGroup.RED

        return ModelInfo(
            model="Llama-3.2-Vision",
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
        """Load and return the Gemma3 multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Gemma3 model instance for multimodal modeling.
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
        print("model", model)
        print("model.config", model.config)
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the Gemma3 multimodal model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
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
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )
        print("inputs", inputs)
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (8, 4)
        else:
            mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution.

        Args:
            model: The MllamaForConditionalGeneration model instance.

        Returns:
            dict: Dictionary mapping model parameters to their sharding specification,
                  or None if tensor parallelism is not needed for this variant.
        """

        shard_specs = {}

        vision_model = model.model.vision_model
        for layer in list(vision_model.transformer.layers) + list(
            vision_model.global_transformer.layers
        ):
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
            shard_specs[layer.mlp.fc2.weight] = ("batch", "model")

        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            attn = getattr(layer, "self_attn", None) or layer.cross_attn
            shard_specs[attn.q_proj.weight] = ("model", "batch")
            shard_specs[attn.k_proj.weight] = ("model", "batch")
            shard_specs[attn.v_proj.weight] = ("model", "batch")
            shard_specs[attn.o_proj.weight] = ("batch", "model")

        return shard_specs
