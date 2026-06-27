# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 model loader implementation for image to text.

Qwen3.6-27B is a multimodal (image-text-to-text) VLM built on the
``Qwen3_5ForConditionalGeneration`` architecture (model_type ``qwen3_5``):

  * a vision tower (``Qwen3_5VisionModel``, SigLIP-style ViT with a conv-based
    patch embed that folds in ``temporal_patch_size``), and
  * a hybrid text decoder (``Qwen3_5TextModel``) that interleaves Gated DeltaNet
    linear-attention layers with standard full-attention layers
    (``full_attention_interval=4`` → 3 linear + 1 full, repeated).

The loader exposes the whole conditional-generation model for an end-to-end
forward, plus an optional ``num_layers`` knob that truncates the text decoder so
the device path can be probed component-by-component without materializing the
full ~55 GB checkpoint.
"""
import torch
from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
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
    """Available Qwen 3.6 model variants for image to text."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_6_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-27B",
            max_length=128,
        ),
    }

    # Default variant to use
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
            num_layers: Optional override for the text decoder depth. When set,
                     only the first ``num_layers`` decoder layers are built —
                     used to probe the device path on a reduced model without
                     loading the full checkpoint.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

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
            model="qwen_v3_6",
            variant=variant,
            group=ModelGroup.GENERALITY,
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
        """Load and return the Qwen 3.6 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its checkpoint dtype (bf16).

        Returns:
            torch.nn.Module: The Qwen 3.6 model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            # Honor the checkpoint's native dtype (bf16) instead of upcasting to fp32.
            model_kwargs["torch_dtype"] = "auto"

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.text_config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output doesn't carry a dynamic
        # cache the pytree comparator can't diff against the CPU golden — same
        # pattern as the qwen_2_5_vl / qwen_3_5 loaders.
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.6 model.

        Args:
            dtype_override: Optional torch.dtype to cast floating-point inputs
                           (e.g. pixel_values) to.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
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

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and torch.is_floating_point(
                    inputs[key]
                ):
                    inputs[key] = inputs[key].to(dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
