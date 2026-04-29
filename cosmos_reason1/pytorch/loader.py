# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Reason1 model loader implementation for vision-language reasoning tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional


from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


def _patch_qwen2_5_vl_for_tt_device():
    """Patch Qwen2.5 VL methods for TT device compatibility.

    Two classes of issues are fixed:

    1. .tolist() on TT tensors: get_rope_index, get_image_features, rot_pos_emb,
       and get_window_index call .tolist() on grid_thw / input_ids. TT device
       does not support eager tensor reads, so these tensors are moved to CPU
       before the call.

    2. torch.repeat_interleave tile-padding: the vision transformer forward uses
       repeat_interleave to build cu_seqlens. On TT device the output VALUE is
       tile-padded (e.g. 2204→2208), corrupting split_with_sizes in the vision
       attention. The fix clamps cu_seqlens to hidden_states.shape[0] inside
       Qwen2_5_VLVisionAttention.forward at the point of use, before
       lengths.tolist() reads the value back to Python.
    """
    try:
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    except ImportError:
        return

    orig_get_rope = modeling_qwen2_5_vl.Qwen2_5_VLModel.get_rope_index
    orig_get_image = modeling_qwen2_5_vl.Qwen2_5_VLModel.get_image_features
    orig_rot_pos = modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb
    orig_get_window = modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.get_window_index
    orig_vis_attn_fwd = modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward

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
            input_ids=input_ids.cpu() if input_ids is not None else None,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
            **kwargs,
        )
        if orig_device is not None:
            position_ids = position_ids.to(orig_device)
            rope_deltas = rope_deltas.to(orig_device)
        return position_ids, rope_deltas

    def _patched_get_image(self, pixel_values, image_grid_thw=None, **kwargs):
        return orig_get_image(
            self,
            pixel_values,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            **kwargs,
        )

    def _patched_rot_pos(self, grid_thw):
        return orig_rot_pos(self, grid_thw.cpu() if grid_thw is not None else grid_thw)

    def _patched_get_window(self, grid_thw):
        return orig_get_window(self, grid_thw.cpu() if grid_thw is not None else grid_thw)

    def _patched_vis_attn_fwd(
        self,
        hidden_states,
        cu_seqlens=None,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        if cu_seqlens is not None:
            # TT repeat_interleave tile-pads tensor VALUES (e.g. 2204 → 2208),
            # corrupting split_with_sizes. Clamp to actual seq len before
            # lengths.tolist() reads the value back to Python.
            cu_seqlens = torch.clamp(cu_seqlens, max=hidden_states.shape[0])
        return orig_vis_attn_fwd(
            self,
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    modeling_qwen2_5_vl.Qwen2_5_VLModel.get_rope_index = _patched_get_rope
    modeling_qwen2_5_vl.Qwen2_5_VLModel.get_image_features = _patched_get_image
    modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb = _patched_rot_pos
    modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.get_window_index = _patched_get_window
    modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward = _patched_vis_attn_fwd


class ModelVariant(StrEnum):
    """Available Cosmos Reason1 model variants for vision-language reasoning tasks."""

    COSMOS_REASON1_7B = "7B"


class ModelLoader(ForgeModel):
    """Cosmos Reason1 model loader implementation for vision-language reasoning tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.COSMOS_REASON1_7B: LLMModelConfig(
            pretrained_model_name="nvidia/Cosmos-Reason1-7B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.COSMOS_REASON1_7B

    # Shared configuration parameters
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

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

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
            model="Cosmos-Reason1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Initialize processor with vision parameters
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cosmos Reason1 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped Cosmos Reason1 model instance for vision-language reasoning tasks.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        _patch_qwen2_5_vl_for_tt_device()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        # transformers 5.x no longer accepts use_cache in __init__; set via config
        model.config.text_config.use_cache = False
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Cosmos Reason1 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Apply chat template to get text prompt
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(self.messages)

        # Process all inputs together
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Convert pixel_values to specified dtype if provided
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
