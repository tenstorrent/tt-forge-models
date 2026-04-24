# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 8B Thinking GGUF model loader implementation for image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.utils.import_utils import (
    PACKAGE_DISTRIBUTION_MAPPING,
    is_gguf_available,
)
from typing import Optional


def _ensure_gguf_detectable():
    if "gguf" not in PACKAGE_DISTRIBUTION_MAPPING:
        PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
    is_gguf_available.cache_clear()


def _patch_qwen3vl_support():
    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
                "qwen3vl"
            ] = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"]
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUF_TO_FAST_CONVERTERS["qwen3"]


def _patched_load_gguf_checkpoint(*args, **kwargs):
    _ensure_gguf_detectable()
    _patch_qwen3vl_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    cfg = result.get("config", {})
    if cfg.get("model_type") == "qwen3vl":
        cfg["model_type"] = "qwen3_vl"
        hidden_size = cfg.get("hidden_size")
        if hidden_size and "vision_config" not in cfg:
            cfg["vision_config"] = {"out_hidden_size": hidden_size}
        elif hidden_size and isinstance(cfg.get("vision_config"), dict):
            cfg["vision_config"].setdefault("out_hidden_size", hidden_size)
    return result


_ensure_gguf_detectable()
_patch_qwen3vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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


def _patch_vision_encoder(model):
    """Work around TT XLA compile-only mode issues in Qwen3 VL.

    In compile-only mode every XLA tensor carries zero data.  The vision-
    language preprocessing is heavily data-dependent (grid_thw loops,
    image-token masks, 3-D position IDs, deepstack) and cannot produce
    correct shapes from zero inputs.

    Fix: split Qwen3VLModel.forward into a disabled preprocessing step
    (runs eagerly — handles vision features, masks, position IDs) and
    the language-model call (remains traceable so dynamo compiles it).
    On the XLA pass the preprocessor returns zero-filled tensors with
    the shapes cached from the CPU pass.
    """
    import transformers.models.qwen3_vl.modeling_qwen3_vl as _qwen3vl_mod

    _cache = {}

    _orig_model_forward = _qwen3vl_mod.Qwen3VLModel.forward

    @torch._dynamo.disable
    def _preprocess(
        mdl,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        cache_position,
        kwargs,
    ):
        """Run the original forward up to (but not including) the
        language-model call.  Returns (inputs_embeds, position_ids,
        attention_mask, visual_pos_masks, deepstack_visual_embeds)."""
        device = "cpu"
        for v in [input_ids, inputs_embeds, attention_mask]:
            if v is not None and isinstance(v, torch.Tensor):
                device = v.device
                break
        is_xla = str(device).startswith("xla")

        if is_xla and "pre" in _cache:
            cached = _cache["pre"]
            out = []
            for item in cached:
                if item is None:
                    out.append(None)
                else:
                    shape, dtype = item
                    out.append(torch.zeros(shape, dtype=dtype, device=device))
            return tuple(out)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = mdl.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_outputs = mdl.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = mdl.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = mdl.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = mdl.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                embed_joint = img_embed.new_zeros(
                    visual_pos_masks.sum(), img_embed.shape[-1]
                ).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = mdl.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        if not is_xla:
            pre = []
            pre.append(
                (inputs_embeds.shape, inputs_embeds.dtype)
                if inputs_embeds is not None
                else None
            )
            pre.append(
                (position_ids.shape, position_ids.dtype)
                if position_ids is not None
                else None
            )
            pre.append(None)  # visual_pos_masks
            pre.append(None)  # deepstack_visual_embeds
            _cache["pre"] = pre

        return (
            inputs_embeds,
            position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        )

    def _patched_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs
    ):
        (
            inputs_embeds,
            position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = _preprocess(
            self,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            cache_position,
            kwargs,
        )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return _qwen3vl_mod.Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

    _qwen3vl_mod.Qwen3VLModel.forward = _patched_model_forward


class ModelVariant(StrEnum):
    """Available Qwen 3 VL 8B Thinking GGUF model variants for image to text."""

    QWEN_3_VL_8B_THINKING_1M_GGUF = "8b_thinking_1m_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 8B Thinking GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_8B_THINKING_1M_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-8B-Thinking-1M-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_8B_THINKING_1M_GGUF

    GGUF_FILE = "Qwen3-VL-8B-Thinking-1M-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 8B Thinking GGUF",
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
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # Re-apply patches right before from_pretrained to ensure they are
        # the outermost wrappers (other GGUF loaders may overwrite them
        # during test collection).
        _ensure_gguf_detectable()
        _patch_qwen3vl_support()
        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        _patch_vision_encoder(model)

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
