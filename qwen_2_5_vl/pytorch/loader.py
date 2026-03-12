# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 VL model loader implementation for vision-language tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AwqConfig, AutoConfig
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
    """Available Qwen 2.5 VL model variants for vision-language tasks."""

    QWEN_2_5_VL_3B_INSTRUCT = "3b_instruct"
    QWEN_2_5_VL_7B_INSTRUCT = "7b_instruct"
    QWEN_2_5_VL_3B_INSTRUCT_AWQ = "3b_instruct_awq"
    QWEN_2_5_VL_7B_INSTRUCT_AWQ = "7b_instruct_awq"
    QWEN_2_5_VL_72B_INSTRUCT = "72b_instruct"


class ModelLoader(ForgeModel):
    """Qwen 2.5 VL model loader implementation for vision-language tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_2_5_VL_3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        ),
        ModelVariant.QWEN_2_5_VL_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        ),
        ModelVariant.QWEN_2_5_VL_3B_INSTRUCT_AWQ: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        ),
        ModelVariant.QWEN_2_5_VL_7B_INSTRUCT_AWQ: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        ),
        ModelVariant.QWEN_2_5_VL_72B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-VL-72B-Instruct",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_VL_3B_INSTRUCT

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

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="qwen_2_5_vl",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.QWEN_2_5_VL_3B_INSTRUCT
            else ModelGroup.GENERALITY,
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

    def load_model(self, dtype_override=None):
        """Load and return the Qwen 2.5 VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped Qwen 2.5 VL model instance for vision-language tasks.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False}

        # Check if this is an AWQ variant and configure accordingly
        if pretrained_model_name in [
            "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        ]:
            quantization_config = AwqConfig(version="ipex")
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "cpu"

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen 2.5 VL model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Apply chat template to get text prompt
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
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

        # Convert pixel_values to specified dtype if provided
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs


    def load_config(self):
        """Load and return the configuration for the Qwen 2.5 VL model variant."""
        self.config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
        return self.config


    def get_mesh_config(self, num_devices: int):
        if not hasattr(self, "config") or self.config is None:
            self.load_config()

        # Handle Qwen2.5-VL config structure (heads count is often in text_config)
        num_heads = getattr(self.config, "num_attention_heads", 0)
        if num_heads == 0 and hasattr(self.config, "text_config"):
            num_heads = getattr(self.config.text_config, "num_attention_heads", 0)

        # Default fallback if heads cannot be determined
        if num_heads == 0:
            return (1, num_devices), ("batch", "model")

        # Prefer (1, N) when heads divide N, otherwise try (2, N/2)
        if num_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif num_heads % (num_devices // 2) == 0 and num_devices % 2 == 0:
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {num_heads} heads across {num_devices} devices"
            )
        mesh_shape = (4, num_devices // 4)
        return mesh_shape, ("batch", "model")


    def load_shard_spec(self, model):
        shard_specs = {}
        # model is usually Wrapper(Qwen2_5_VLForConditionalGeneration)
        # We access the underlying HF model via .model
        root = model.model

        # --- Text Model Sharding ---
        # The text backbone is located at root.model.language_model
        text_layers = root.model.language_model.layers
        for layer in text_layers:
            # MLP
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            # Attention
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            #shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            #shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            #shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            #shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            #shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        # LM Head
        shard_specs[root.lm_head.weight] = ("model", "batch")

        # --- Vision Model Sharding ---
        # The vision backbone is located at root.model.visual
        visual = root.model.visual

        # Patch Embedding
        # Conv3d(3, 1280, ...) -> Split output channels (dim 0)
        #shard_specs[visual.patch_embed.proj.weight] = ("model", "batch", None, None, None)
        #if visual.patch_embed.proj.bias is not None:
        #    shard_specs[visual.patch_embed.proj.bias] = ("model",)

        # Vision Blocks
        '''for block in visual.blocks:
            # MLP
            shard_specs[block.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[block.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[block.mlp.down_proj.weight] = ("batch", "model")

            # Attention
            # qkv is a fused Linear(1280, 3840) -> Colwise split is safe
            shard_specs[block.attn.qkv.weight] = ("model", "batch")
            #if block.attn.qkv.bias is not None:
            #    shard_specs[block.attn.qkv.bias] = ("model",)
            # proj is Linear(1280, 1280) -> Rowwise split
            shard_specs[block.attn.proj.weight] = ("batch", "model")'''

        # Vision Merger (Sequential MLP: Linear -> GELU -> Linear)
        # merger.mlp[0]: Linear(5120->5120) -> Colwise
        #shard_specs[visual.merger.mlp[0].weight] = ("model", "batch")
        #if visual.merger.mlp[0].bias is not None:
        #    shard_specs[visual.merger.mlp[0].bias] = ("model",)

        # merger.mlp[2]: Linear(5120->2048) -> Rowwise
        #shard_specs[visual.merger.mlp[2].weight] = ("batch", "model")

        return shard_specs
