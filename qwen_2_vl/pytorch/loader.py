# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2 VL model loader implementation for vision-language tasks.
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AwqConfig
from typing import Optional


from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available Qwen 2 VL model variants for vision-language tasks."""

    QWEN_2_VL_2B_INSTRUCT = "2B_Instruct"
    QWEN_2_VL_7B = "7B"
    QWEN_2_VL_7B_INSTRUCT = "7B_Instruct"
    QWEN_2_VL_2B_INSTRUCT_AWQ = "2B_INSTRUCT_Awq"
    QWEN_2_VL_2B_INSTRUCT_GPTQ_INT4 = "2B_INSTRUCT_Gptq_Int4"

    # mlx-community quantized variants
    QWEN_2_VL_2B_INSTRUCT_4BIT = "2B_Instruct_4bit"


class ModelLoader(ForgeModel):
    """Qwen 2 VL model loader implementation for vision-language tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_2_VL_2B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2-VL-2B-Instruct",
        ),
        ModelVariant.QWEN_2_VL_7B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2-VL-7B",
        ),
        ModelVariant.QWEN_2_VL_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2-VL-7B-Instruct",
        ),
        # mlx-community quantized variants
        ModelVariant.QWEN_2_VL_2B_INSTRUCT_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_2_VL_2B_INSTRUCT

    # Shared configuration parameters
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.raw_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 2-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Initialize processor with vision parameters
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 2 VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped Qwen 2 VL model instance for vision-language tasks.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        # Check if this is a quantized variant and configure accordingly
        if pretrained_model_name in [
            "Qwen/Qwen2-VL-2B-Instruct-AWQ",
        ]:
            quantization_config = AwqConfig(version="ipex")
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "cpu"
        elif pretrained_model_name in [
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        ]:
            model_kwargs["device_map"] = "cpu"

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32

        if "mlx-community" in pretrained_model_name:
            model_kwargs["ignore_mismatched_sizes"] = True

        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.raw_model = model
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen 2 VL model with this instance's variant settings.

        Pre-computes vision embeddings on CPU so the vision encoder
        does not need to go through torch.compile/dynamo.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts floating-point tensors to the specified dtype.

        Returns:
            dict: Input tensors (inputs_embeds, attention_mask, position_ids)
                  that can be fed directly to the language model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Flatten content items from messages for chat template compatibility
        content_items = [
            item
            for msg in self.messages
            for item in (
                msg["content"] if isinstance(msg["content"], list) else [msg["content"]]
            )
        ]
        text = self.processor.apply_chat_template(
            content_items, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(self.messages)

        # Process all inputs together
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Pre-compute vision embeddings on CPU to avoid dynamo issues
        # with the vision encoder's dynamic rotary position embeddings.
        model = self.raw_model
        with torch.no_grad():
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            image_embeds = model.model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            ).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Build position_ids, passing mm_token_type_ids if supported
            import inspect

            pos_kwargs = {
                "input_ids": input_ids,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": None,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            sig = inspect.signature(model.model.compute_3d_position_ids)
            if "mm_token_type_ids" in sig.parameters:
                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
                mm_token_type_ids[input_ids == model.config.image_token_id] = 1
                pos_kwargs["mm_token_type_ids"] = mm_token_type_ids

            position_ids = model.model.compute_3d_position_ids(**pos_kwargs)

        result = {
            "inputs_embeds": inputs_embeds.detach(),
            "attention_mask": attention_mask,
            "position_ids": position_ids.detach(),
        }

        if dtype_override is not None:
            result["inputs_embeds"] = result["inputs_embeds"].to(dtype_override)

        return result
