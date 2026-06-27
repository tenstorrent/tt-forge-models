# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.6-27B vision-tower loader (image encoder only).

The Qwen3.6 VLM is multimodal; per the model-bringup multimodal contract the
vision tower is brought up as its own component. This loader extracts the
``Qwen3_5VisionModel`` (patch-16 ViT-style encoder, hidden 1152, depth 27,
16 heads, spatial_merge 2 → out_hidden 5120) from the full conditional model and
exposes it as a single forward pass over patchified image features.

The vision tower is small (fits a single chip), unlike the ~27B hybrid
Gated-DeltaNet text decoder (see the ``causal_lm`` sibling loader).
"""

from typing import Optional

import torch
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


class _VisionTower(torch.nn.Module):
    """Thin wrapper so the runner can call the vision tower as model(**inputs).

    ``Qwen3_5VisionModel.forward(hidden_states, grid_thw)`` consumes the
    patchified image features the processor emits as ``pixel_values`` together
    with ``image_grid_thw``. Expose those exact names so a kwargs dict drives it.
    """

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values, image_grid_thw):
        return self.visual(pixel_values, image_grid_thw)


class ModelVariant(StrEnum):
    """Available Qwen3.6 vision-tower variants."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen3.6-27B vision-tower (image encoder) loader."""

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
    sample_prompt = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3.6 vision tower",
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
        """Load the full VLM, extract and return the vision tower only.

        The text decoder is dropped after extraction to free host memory; only
        the (small) vision tower is kept and brought up on device.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["dtype"] = "auto"
        model_kwargs |= kwargs

        full = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        visual = full.model.visual
        # Drop the heavy text decoder + lm_head; keep only the vision tower.
        del full.model.language_model
        del full.lm_head

        wrapper = _VisionTower(visual).eval()
        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return patchified image features + grid for the vision tower.

        Keys (pixel_values, image_grid_thw) match _VisionTower.forward.
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
        proc = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        pixel_values = proc["pixel_values"]
        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": proc["image_grid_thw"],
        }
