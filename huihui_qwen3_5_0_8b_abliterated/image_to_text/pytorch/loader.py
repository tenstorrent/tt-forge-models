# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen3.5-0.8B abliterated model loader implementation for image to text.
"""

from transformers import AutoModelForImageTextToText, AutoProcessor
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


def _patch_qwen3_5_for_tt_device():
    """Patch Qwen3.5 VL methods that call .tolist() or boolean-index on device tensors.

    TT device does not support eager tensor reads — any .tolist() or boolean
    indexing on a TT tensor triggers a device sync that fails with Error code: 13.
    Move the small integer metadata tensors (grid_thw, input_ids) and tensors
    involved in boolean-index checks to CPU; move computed outputs (position_ids,
    rope_deltas, masks) back to the original device so the LLM path stays on TT.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        return

    orig_fast_pos = modeling_qwen3_5.Qwen3_5VisionModel.fast_pos_embed_interpolate
    orig_rot_pos = modeling_qwen3_5.Qwen3_5VisionModel.rot_pos_emb
    orig_get_rope = modeling_qwen3_5.Qwen3_5Model.get_rope_index
    orig_get_image = modeling_qwen3_5.Qwen3_5Model.get_image_features
    orig_get_placeholder_mask = modeling_qwen3_5.Qwen3_5Model.get_placeholder_mask

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
        position_ids, rope_deltas = orig_get_rope(
            self,
            input_ids=input_ids.cpu() if input_ids is not None else None,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
            **kwargs,
        )
        # input_ids is int64 and may stay on CPU in the TT backend; do NOT
        # use input_ids.device as the target — use the model's float parameter
        # device (TT/XLA) so position_ids and rope_deltas land on the correct
        # device for the language model attention computation.
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = None
        if target_device is not None:
            position_ids = position_ids.to(target_device)
            rope_deltas = rope_deltas.to(target_device)
        return position_ids, rope_deltas

    def _patched_get_image(self, pixel_values, image_grid_thw=None, **kwargs):
        return orig_get_image(
            self,
            pixel_values,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            **kwargs,
        )

    def _patched_get_placeholder_mask(
        self,
        input_ids,
        inputs_embeds,
        image_features=None,
        video_features=None,
    ):
        # The original get_placeholder_mask checks token/feature count parity via
        # inputs_embeds[special_image_mask].numel(), which requires boolean indexing
        # on TT device tensors — an unsupported op (Error code: 13).  Replace that
        # check with the equivalent arithmetic:
        #   inputs_embeds[bool_mask].numel() == n_image_tokens * hidden_size
        # where n_image_tokens is computed on CPU (input_ids is always CPU int64).
        import torch
        from transformers.utils import torch_compilable_check

        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.image_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.video_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        hidden_size = inputs_embeds.shape[-1]

        n_image_tokens = special_image_mask.sum()
        special_image_mask = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if image_features is not None:
            torch_compilable_check(
                int(n_image_tokens.item()) * hidden_size == image_features.numel(),
                f"Image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features shape={image_features.shape}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = (
            special_video_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if video_features is not None:
            torch_compilable_check(
                int(n_video_tokens.item()) * hidden_size == video_features.numel(),
                f"Video features and video tokens do not match: "
                f"tokens={n_video_tokens}, features shape={video_features.shape}",
            )

        return special_image_mask, special_video_mask

    modeling_qwen3_5.Qwen3_5VisionModel.fast_pos_embed_interpolate = _patched_fast_pos
    modeling_qwen3_5.Qwen3_5VisionModel.rot_pos_emb = _patched_rot_pos
    modeling_qwen3_5.Qwen3_5Model.get_rope_index = _patched_get_rope
    modeling_qwen3_5.Qwen3_5Model.get_image_features = _patched_get_image
    modeling_qwen3_5.Qwen3_5Model.get_placeholder_mask = _patched_get_placeholder_mask


class ModelVariant(StrEnum):
    """Available Huihui Qwen3.5-0.8B abliterated model variants for image to text."""

    QWEN_3_5_0_8B_ABLITERATED = "0_8b_abliterated"


class ModelLoader(ForgeModel):
    """Huihui Qwen3.5-0.8B abliterated model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_0_8B_ABLITERATED: LLMModelConfig(
            pretrained_model_name="huihui-ai/Huihui-Qwen3.5-0.8B-abliterated",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_0_8B_ABLITERATED

    # Standard pixel limits for Qwen VL models to stay within hardware L1 budget
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="huihui_qwen3_5_0_8b_abliterated",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.processor.image_processor.min_pixels = self.min_pixels
        self.processor.image_processor.max_pixels = self.max_pixels

        _patch_qwen3_5_for_tt_device()

        model = AutoModelForImageTextToText.from_pretrained(
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
        inputs["use_cache"] = False
        return inputs
