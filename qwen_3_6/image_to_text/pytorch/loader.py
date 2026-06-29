# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 model loader implementation for image to text.

Qwen3.6-35B-A3B is a Mixture-of-Experts vision-language model
(``Qwen3_5MoeForConditionalGeneration``) pairing a SigLIP-style vision tower
with a ``qwen3_5_moe`` text decoder. The decoder uses 256 routed experts
(8 active per token) and a hybrid Gated DeltaNet (linear attention with a
causal conv) + full-attention layout, the same text stack as ``qwen_3_5``.
"""
from transformers import Qwen3_5MoeForConditionalGeneration, AutoProcessor, AutoConfig
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

    QWEN_3_6_35B_A3B = "35b_a3b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_6_35B_A3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-35B-A3B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_35B_A3B

    # Shared configuration parameters
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional override for the number of text-decoder layers,
                     used for reduced-layer device/op probing.
        """
        super().__init__(variant)
        self.processor = None
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

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.6 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 3.6 model instance for image to text.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.text_config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, dtype="auto", **model_kwargs
        )
        model.eval()

        # Avoid emitting a DynamicCache in the forward output, which the
        # runner's pytree comparator can't diff leaf-wise against the golden.
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.6 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.sample_image},
                    {"type": "text", "text": "Describe this image."},
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
