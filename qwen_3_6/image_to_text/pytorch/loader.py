# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 model loader implementation for image-to-text (vision-language).

Qwen3.6-35B-A3B is a Qwen3_5MoeForConditionalGeneration multimodal model:
  * a SigLIP-style vision tower (Conv3d patch embed, temporal_patch_size=2,
    27-layer encoder, hidden 1152) that produces image embeddings, and
  * a Qwen3.5-MoE text decoder (hidden 2048, 40 layers, 256 experts / top-8)
    whose layers interleave Gated DeltaNet linear attention with full
    attention (3x linear_attention + 1x full_attention repeated).

Total ~35B params, ~3B active per token ("A3B"). Both components are brought
up on device separately (see the bringup report's component-support table).
"""
import torch
from transformers import (
    Qwen3_5MoeForConditionalGeneration,
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
    """Available Qwen 3.6 MoE VLM variants for image to text."""

    QWEN_3_6_35B_A3B = "35b_a3b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 MoE VLM loader for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_35B_A3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-35B-A3B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_35B_A3B

    # Sample image used for the multimodal input.
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader.

        Args:
            variant: Optional ModelVariant; DEFAULT_VARIANT if None.
            num_layers: Optional override of the text decoder depth, for
                reduced-layer device probing of the MoE / GatedDeltaNet stack.
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
        """Load and return the Qwen 3.6 MoE VLM instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model dtype.

        Returns:
            torch.nn.Module: The Qwen3_5MoeForConditionalGeneration instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.text_config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output is not a dynamic cache
        # object the runner's pytree comparator can't diff leaf-wise against
        # the CPU golden (same pattern as qwen_3_5 / qwen_2_5_vl loaders).
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample multimodal inputs for the Qwen 3.6 VLM.

        Returns:
            dict: input_ids / attention_mask / pixel_values / image_grid_thw.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

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

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
