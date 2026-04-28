# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Reason2 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
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


def _patch_qwen3vl_for_tt_device():
    """Patch Qwen3VL methods that call .tolist() on TT device tensors.

    The test runner moves all input tensors to TT device, but VisionModel's
    fast_pos_embed_interpolate and rot_pos_emb, and Qwen3VLModel's get_rope_index
    call .tolist() on grid_thw / input_ids tensors for Python control flow.
    TT device does not support eager tensor reads; .tolist() triggers a device
    sync that fails with Error code: 13. Moving these tensors to CPU before the
    .tolist() calls avoids the sync while keeping all actual computations on TT
    device.
    """
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except ImportError:
        return

    orig_fast_pos = modeling_qwen3_vl.Qwen3VLVisionModel.fast_pos_embed_interpolate
    orig_rot_pos = modeling_qwen3_vl.Qwen3VLVisionModel.rot_pos_emb
    orig_get_rope = modeling_qwen3_vl.Qwen3VLModel.get_rope_index

    def _patched_fast_pos(self, grid_thw):
        return orig_fast_pos(self, grid_thw.cpu())

    def _patched_rot_pos(self, grid_thw):
        return orig_rot_pos(self, grid_thw.cpu())

    def _patched_get_rope(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        orig_device = input_ids.device if input_ids is not None else None
        position_ids, rope_deltas = orig_get_rope(
            self,
            input_ids.cpu() if input_ids is not None else None,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
            **kwargs,
        )
        if orig_device is not None and orig_device.type != "cpu":
            position_ids = position_ids.to(orig_device)
            rope_deltas = rope_deltas.to(orig_device)
        return position_ids, rope_deltas

    modeling_qwen3_vl.Qwen3VLVisionModel.fast_pos_embed_interpolate = _patched_fast_pos
    modeling_qwen3_vl.Qwen3VLVisionModel.rot_pos_emb = _patched_rot_pos
    modeling_qwen3_vl.Qwen3VLModel.get_rope_index = _patched_get_rope


class ModelVariant(StrEnum):
    """Available Cosmos Reason2 GGUF model variants for image to text."""

    COSMOS_REASON2_2B_GGUF = "2b_gguf"


class ModelLoader(ForgeModel):
    """Cosmos Reason2 GGUF model loader implementation for image to text tasks."""

    # nvidia/Cosmos-Reason2-2B is a gated repo; use public Qwen3-VL-2B-Instruct (same architecture)
    BASE_QWEN3_VL_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

    _VARIANTS = {
        ModelVariant.COSMOS_REASON2_2B_GGUF: LLMModelConfig(
            pretrained_model_name="apolo13x/Cosmos-Reason2-2B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COSMOS_REASON2_2B_GGUF

    GGUF_FILE = "Cosmos-Reason2-2B-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Cosmos Reason2 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _patch_qwen3vl_for_tt_device()

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # nvidia/Cosmos-Reason2-2B is gated; Cosmos-Reason2-2B is fine-tuned from
        # Qwen3-VL-2B-Instruct and shares the same processor.
        self.processor = AutoProcessor.from_pretrained(self.BASE_QWEN3_VL_MODEL)
        self.processor.image_processor.min_pixels = 56 * 56
        self.processor.image_processor.max_pixels = 13 * 28 * 1280

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
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

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
