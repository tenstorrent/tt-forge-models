# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 model loader implementation for image-to-text (vision-language).

Qwen 3.6 (HuggingFace ``model_type`` ``qwen3_5``, architecture
``Qwen3_5ForConditionalGeneration``) is a vision-language model pairing a
SigLIP-style vision tower (Conv3d patch-embed, temporal_patch_size=2,
spatial_merge_size=2) with the Qwen 3.5 hybrid text decoder. The decoder
interleaves Gated DeltaNet linear-attention layers (causal conv1d + chunked
delta rule, SSM state in fp32) with a full-attention layer every 4th block
(``full_attention_interval=4``) and uses interleaved M-RoPE for image/text
positions.
"""
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen 3.6 model variants for image-to-text."""

    QWEN_3_6_27B = "27B"


class ModelLoader(ForgeModel):
    """Qwen 3.6 model loader implementation for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-27B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_27B

    # Shared configuration parameters
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    sample_text = "Describe this image."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional override for the text decoder depth (for
                     layer-reduced hardware bring-up of this 27B model).
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen 3.6",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.6 VLM instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 3.6 vision-language model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            # Qwen 3.6 keeps the decoder depth in the nested text_config; setting
            # it on the outer config is ignored. Set text_config and truncate
            # layer_types so the hybrid linear/full pattern still includes a
            # full_attention layer (same handling as the qwen_3_5 causal_lm loader).
            config = AutoConfig.from_pretrained(pretrained_model_name)
            text_cfg = getattr(config, "text_config", config)
            text_cfg.num_hidden_layers = self.num_layers
            if getattr(text_cfg, "layer_types", None) is not None:
                text_cfg.layer_types = text_cfg.layer_types[: self.num_layers]
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False on the live text_config so the forward output
        # does not include a dynamic cache, which the runner's pytree comparator
        # can't diff leaf-wise against the CPU golden (same pattern as the
        # qwen_2_5_vl / qwen_3_5 loaders).
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        model.config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.6 VLM.

        Args:
            dtype_override: Optional torch.dtype to cast pixel_values.
            batch_size: Batch size (only 1 is validated for this VLM).

        Returns:
            dict: Input tensors (input_ids, attention_mask, pixel_values,
                  image_grid_thw) that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.sample_image},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
