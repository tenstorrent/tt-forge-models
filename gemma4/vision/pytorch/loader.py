# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 vision-tower loader.

google/gemma-4-26B-A4B-it (Gemma4ForConditionalGeneration) pairs a sparse-MoE
text decoder with a SigLIP-style vision tower (``Gemma4VisionModel``). This
loader brings up that vision tower in isolation — the image-encoder op
pre-check for the multimodal model — driving a single forward pass over the
pre-patchified pixel values the Gemma4 processor produces.

The processor emits ``pixel_values`` of shape ``[batch, num_patches, patch_dim]``
(NaFlex-style variable-resolution patchification, patch_dim = 16*16*3 = 768) and
``image_position_ids`` of shape ``[batch, num_patches, 2]`` giving each patch's
(x, y) grid coordinate (padding patches are (-1, -1)). The vision tower's forward
signature is ``forward(pixel_values, pixel_position_ids)``.
"""

from typing import Optional

from transformers import AutoConfig, AutoProcessor, Gemma4ForConditionalGeneration

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
from ....tools.utils import cast_input_to_type, get_file
from PIL import Image


class ModelVariant(StrEnum):
    """Available Gemma4 vision-tower variants."""

    GEMMA_4_26B_A4B_IT = "26B-A4B-it"


class ModelLoader(ForgeModel):
    """Gemma4 vision-tower (image encoder) loader."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_26B_A4B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_26B_A4B_IT

    sample_text = "Describe this image."
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Gemma 4 vision tower",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the full Gemma4 multimodal model and return its vision tower.

        Only the ``Gemma4VisionModel`` submodule is returned, so the compiled
        graph covers the image encoder alone (the text decoder is brought up
        separately in ``gemma4/pytorch``).

        Args:
            dtype_override: Optional torch dtype to load weights in. The model
                ships in bfloat16.

        Returns:
            torch.nn.Module: The Gemma4 vision tower (image encoder).
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        full_model = Gemma4ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        full_model.eval()
        self.config = full_model.config
        # Gemma4ForConditionalGeneration.model is Gemma4Model, which holds the
        # vision tower alongside the language model.
        vision_tower = full_model.model.vision_tower
        vision_tower.eval()
        self.model = vision_tower
        return vision_tower

    def load_inputs(self, dtype_override=None, image_url: Optional[str] = None):
        """Build vision-tower inputs from a sample image via the processor.

        Returns:
            dict: {"pixel_values", "pixel_position_ids"} for Gemma4VisionModel.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.sample_text},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        proc_out = self.processor(
            text=text_prompt, images=[image], return_tensors="pt"
        )

        pixel_values = proc_out["pixel_values"]
        # processor names the per-patch (x, y) grid positions image_position_ids;
        # the vision tower's forward parameter is pixel_position_ids.
        pixel_position_ids = proc_out["image_position_ids"]

        if dtype_override is not None:
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        return {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }
