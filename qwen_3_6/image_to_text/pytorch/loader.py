# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 model loader implementation for image-text-to-text (VLM).

Qwen3.6-27B is a vision-language model built on the ``qwen3_5`` architecture
(``Qwen3_5ForConditionalGeneration``):

  * Text decoder (``qwen3_5_text``) is a hybrid stack interleaving Gated DeltaNet
    linear-attention layers with standard full-attention layers
    (``full_attention_interval=4``), i.e. 3x linear_attention + 1x full_attention
    repeated for 64 layers. Linear attention uses a causal conv1d + chunked
    delta rule with ``mamba_ssm_dtype=float32``.
  * Vision tower (``qwen3_5``) is a SigLIP-style ViT (depth 27, hidden 1152,
    patch 16, spatial_merge 2) projecting to the text hidden size (5120).
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
    """Available Qwen 3.6 model variants for image-text-to-text."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-27B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_27B

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
            num_layers: Optional override for the text decoder depth, used to
                        build a layer-reduced model for cheap CPU-time estimation
                        and device op pre-checks.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="qwen_3_6",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.6 VLM instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 3.6 model instance for image-text-to-text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        else:
            model_kwargs["dtype"] = "auto"

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.text_config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # Force use_cache=False so the forward output does not include a
        # DynamicCache, which the runner's pytree comparator can't diff
        # leaf-wise against the CPU golden (same pattern as qwen_3_5 causal_lm).
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample image+text inputs for the Qwen 3.6 VLM.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

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
