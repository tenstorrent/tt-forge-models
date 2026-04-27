# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AxionML Qwen3.5-2B NVFP4 model loader implementation for image to text.
"""

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel, Qwen3_5Model
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


def _patch_qwen35_for_tt_device():
    """Patch Qwen3.5 vision model methods that call .tolist() on TT device tensors.

    TT device tensors do not support .tolist() / Python-side data access.
    These methods use .tolist() only for control-flow (computing grid sizes and
    position indices), so moving those tensors to CPU is safe — the main
    computation (vision encoder, attention) still runs on TT.
    """
    _orig_fast_pos_embed = Qwen3_5VisionModel.fast_pos_embed_interpolate
    _orig_rot_pos_emb = Qwen3_5VisionModel.rot_pos_emb
    _orig_get_image_features = Qwen3_5Model.get_image_features
    _orig_get_rope_index = Qwen3_5Model.get_rope_index

    def _patched_fast_pos_embed(self, grid_thw):
        return _orig_fast_pos_embed(self, grid_thw.cpu())

    def _patched_rot_pos_emb(self, grid_thw):
        return _orig_rot_pos_emb(self, grid_thw.cpu())

    def _patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        image_grid_thw_cpu = image_grid_thw.cpu() if image_grid_thw is not None else None
        return _orig_get_image_features(self, pixel_values, image_grid_thw_cpu, **kwargs)

    def _patched_get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        orig_device = input_ids.device if input_ids is not None else None
        input_ids_cpu = input_ids.cpu() if input_ids is not None else None
        image_grid_thw_cpu = image_grid_thw.cpu() if image_grid_thw is not None else None
        video_grid_thw_cpu = video_grid_thw.cpu() if video_grid_thw is not None else None
        attention_mask_cpu = attention_mask.cpu() if attention_mask is not None else None
        position_ids, rope_deltas = _orig_get_rope_index(
            self,
            input_ids_cpu,
            image_grid_thw_cpu,
            video_grid_thw_cpu,
            attention_mask_cpu,
            **kwargs,
        )
        if orig_device is not None:
            position_ids = position_ids.to(orig_device)
            rope_deltas = rope_deltas.to(orig_device)
        return position_ids, rope_deltas

    Qwen3_5VisionModel.fast_pos_embed_interpolate = _patched_fast_pos_embed
    Qwen3_5VisionModel.rot_pos_emb = _patched_rot_pos_emb
    Qwen3_5Model.get_image_features = _patched_get_image_features
    Qwen3_5Model.get_rope_index = _patched_get_rope_index


class ModelVariant(StrEnum):
    """Available AxionML Qwen3.5-2B NVFP4 model variants for image to text."""

    QWEN_3_5_2B_NVFP4 = "2B_NVFP4"


class ModelLoader(ForgeModel):
    """AxionML Qwen3.5-2B NVFP4 model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_2B_NVFP4: LLMModelConfig(
            pretrained_model_name="AxionML/Qwen3.5-2B-NVFP4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_2B_NVFP4

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
            model="AxionML Qwen3.5-2B NVFP4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AxionML Qwen3.5-2B NVFP4 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The AxionML Qwen3.5-2B NVFP4 model instance for image to text.
        """
        _patch_qwen35_for_tt_device()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["device_map"] = "cpu"
        # AxionML Qwen3.5-2B-NVFP4 uses NVIDIA modelopt NVFP4 quantization.
        # Without nvidia-modelopt installed, transformers creates unquantized
        # layers whose weight shapes are 2x the packed checkpoint shapes.
        # ignore_mismatched_sizes=True bypasses the resulting load error so the
        # model can compile and run on TT hardware.
        model_kwargs["ignore_mismatched_sizes"] = True

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the AxionML Qwen3.5-2B NVFP4 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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
