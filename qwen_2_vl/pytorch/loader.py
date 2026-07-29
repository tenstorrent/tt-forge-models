# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen2-VL model loader implementation for vision-language tasks.
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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
    """Available Qwen2-VL model variants for vision-language tasks."""

    QWEN_2_VL_72B_INSTRUCT = "72B_Instruct"


class ModelLoader(ForgeModel):
    """Qwen2-VL model loader implementation for vision-language tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_2_VL_72B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2-VL-72B-Instruct",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_2_VL_72B_INSTRUCT

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
        self.config = None

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
            model="Qwen2-VL",
            variant=variant,
            group=ModelGroup.GENERALITY,
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
        """Load and return the Qwen2-VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The Wrapped Qwen2-VL model instance for vision-language tasks.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.text_config.use_cache = False
        model.eval()
        self.config = model.config
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen2-VL model with this instance's variant settings.

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

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)

        assert (
            self.config.text_config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard spec for the Qwen2 backbone (GQA, q/k/v bias)."""
        hf = model.model
        text_model = hf.model.language_model
        visual = hf.model.visual

        shard_specs = {text_model.embed_tokens.weight: ("model", "batch")}

        for layer in text_model.layers:
            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", "batch")
            shard_specs[attn.q_proj.bias] = ("model",)
            shard_specs[attn.k_proj.weight] = ("model", "batch")
            shard_specs[attn.k_proj.bias] = ("model",)
            shard_specs[attn.v_proj.weight] = ("model", "batch")
            shard_specs[attn.v_proj.bias] = ("model",)
            shard_specs[attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[hf.lm_head.weight] = ("model", "batch")

        # Vision tower. The ViT uses a *fused* qkv Linear; column-parallel on the
        # fused output + row-parallel on proj stays correct under Shardy.
        for block in visual.blocks:
            attn = block.attn
            shard_specs[attn.qkv.weight] = ("model", "batch")
            shard_specs[attn.qkv.bias] = ("model",)
            shard_specs[attn.proj.weight] = ("batch", "model")
            shard_specs[attn.proj.bias] = ("batch",)

            shard_specs[block.mlp.fc1.weight] = ("model", "batch")
            shard_specs[block.mlp.fc1.bias] = ("model",)
            shard_specs[block.mlp.fc2.weight] = ("batch", "model")
            shard_specs[block.mlp.fc2.bias] = ("batch",)

        # PatchMerger MLP: Sequential(Linear, GELU, Linear) -> column then row.
        up, down = (m for m in visual.merger.mlp if isinstance(m, torch.nn.Linear))
        shard_specs[up.weight] = ("model", "batch")
        shard_specs[up.bias] = ("model",)
        shard_specs[down.weight] = ("batch", "model")
        shard_specs[down.bias] = ("batch",)

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
