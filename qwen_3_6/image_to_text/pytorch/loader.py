# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.6-27B model loader implementation for image-to-text (vision-language).

Qwen3.6-27B is a multimodal VLM exposing the model_type ``qwen3_5``
(``Qwen3_5ForConditionalGeneration``):
  * Vision tower (``qwen3_5`` vision): patch-16 ViT-style encoder, hidden 1152,
    depth 27, 16 heads, spatial_merge 2, projecting to out_hidden 5120.
  * Text decoder (``qwen3_5_text``): dense decoder-only, hidden 5120, 64 layers,
    GQA (24 q : 4 kv, head_dim 256), SwiGLU (intermediate 17408), vocab 248320.

This loader loads the full conditional-generation model. The text decoder alone
is ~27B params (~54 GB bf16) and does not fit a single Blackhole chip; see the
``causal_lm`` sibling loader for the text-decoder-only path used for sharded
device bringup.
"""

from typing import Optional

from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen3.6 VLM variants for image-to-text."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen3.6-27B loader for image-to-text (vision-language) tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-27B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_27B

    # Public sample image used by load_inputs.
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    sample_prompt = "Describe this image."

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
        return ModelInfo(
            model="Qwen 3.6",
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
        """Load and return the Qwen3.6-27B VLM instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. If not provided, the checkpoint dtype (bf16)
                            is used.

        Returns:
            torch.nn.Module: The Qwen3.6-27B conditional-generation model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["dtype"] = "auto"
        model_kwargs |= kwargs

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output does not include a
        # DynamicCache, which the runner's pytree comparator can't diff
        # leaf-wise against the CPU golden (same pattern as qwen_3_5 causal_lm).
        model.config.use_cache = False

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build sample multimodal inputs (one image + a text prompt).

        Returns:
            dict: Processor outputs (input_ids, attention_mask, pixel_values,
                  image_grid_thw, ...) ready for the model forward.
        """
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.sample_image},
                    {"type": "text", "text": self.sample_prompt},
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
