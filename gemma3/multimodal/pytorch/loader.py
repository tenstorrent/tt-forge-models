# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 model loader implementation for multimodal modeling.
"""

from typing import Optional, List, Dict, Any
from PIL import Image

from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
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


class ModelVariant(StrEnum):
    """Available Gemma3 multimodal model variants."""

    GEMMA_3_4B_IT = "google/gemma-3-4b-it"
    GEMMA_3_12B_IT = "google/gemma-3-12b-it"
    GEMMA_3_27B_IT = "google/gemma-3-27b-it"


class ModelLoader(ForgeModel):
    """Gemma3 model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_4B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_3_4B_IT),
            max_length=512,
        ),
        ModelVariant.GEMMA_3_12B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_3_12B_IT),
            max_length=512,
        ),
        ModelVariant.GEMMA_3_27B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_3_27B_IT),
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_4B_IT

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        if any(x in variant.value for x in ["12b", "27b"]):
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="gemma_3_multimodal",
            variant=variant,
            group=group,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the Gemma3 multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            disable_sliding_window: If True, disables sliding window attention to avoid
                                   dynamic shape issues with Shardy propagation.
                                   If None, checks GEMMA3_DISABLE_SLIDING_WINDOW environment variable.

        Returns:
            torch.nn.Module: The Gemma3 model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model = Gemma3ForConditionalGeneration.from_pretrained(
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
        """Load and return sample inputs for the Gemma3 multimodal model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        max_length = self._variant_config.max_length
        if self.processor is None:
            self._load_processor()

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt or self.sample_text},
            ],
        }

        inputs = self.processor.apply_chat_template(
            [message],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )

        if dtype_override is not None:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "token_type_ids": inputs["token_type_ids"],
            "use_cache": False,
        }
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

        if self._variant == ModelVariant.GEMMA_3_27B_IT:
            assert (
                self.config.text_config.num_attention_heads % mesh_shape[1] == 0
            ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution.

        Args:
            model: The Gemma3ForConditionalGeneration model instance.

        Returns:
            dict: Dictionary mapping model parameters to their sharding specification,
                  or None if tensor parallelism is not needed for this variant.
        """
        if self._variant != ModelVariant.GEMMA_3_27B_IT:
            return None

        shard_specs = {}

        for layer in model.vision_tower.vision_model.encoder.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.out_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
            shard_specs[layer.mlp.fc2.weight] = ("batch", "model")

        for layer in model.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        return shard_specs
