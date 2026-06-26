# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 model loader implementation for image-text-to-text (VLM).

Qwen 3.5 VLM pairs a Qwen3-VL-style vision tower (SigLIP-style patch encoder
with spatial/temporal patch merging, out_hidden_size 5120) with a hybrid
Qwen 3.5 text decoder that interleaves Gated DeltaNet (linear attention with
causal conv1d + chunked delta rule) and standard full attention layers
(full_attention_interval=4). The full forward exercises both the vision tower
and the text decoder in a single graph.
"""

from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
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
    """Available Qwen 3.5 VLM variants for image-text-to-text."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen 3.5 model loader implementation for image-text-to-text (VLM) tasks."""

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

    # Cap the number of vision patches so the vision-tower sequence stays small
    # enough to compile (a full-resolution image produces thousands of patches).
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28

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
            model="Qwen 3.5 VLM",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.5 VLM model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its checkpoint dtype (bf16).

        Returns:
            torch.nn.Module: The Qwen 3.5 VLM model instance for image-text-to-text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # Force use_cache=False on the live model config (and the nested text
        # config) so the forward output does not include a hybrid cache object,
        # which the runner's pytree comparator can't diff leaf-wise against the
        # CPU golden. Same pattern as the qwen_3_5 causal_lm loader.
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.5 VLM with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
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
        return inputs
