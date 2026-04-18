# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolVLM2 model loader implementation for image-text-to-text generation.
"""

from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available SmolVLM2 model variants."""

    SMOLVLM2_256M_VIDEO_INSTRUCT_GGUF = "256M_Video_Instruct_GGUF"


class ModelLoader(ForgeModel):
    """SmolVLM2 model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.SMOLVLM2_256M_VIDEO_INSTRUCT_GGUF: ModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOLVLM2_256M_VIDEO_INSTRUCT_GGUF

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SmolVLM2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        model.eval()
        self._model = model

        if self.processor is None:
            self._load_processor()

        return model

    def _precompute_merged_embeds(self, input_ids, pixel_values, pixel_attention_mask):
        """Pre-compute inputs_embeds with image features merged in on CPU.

        This avoids dynamo graph breaks from data-dependent indexing in
        get_image_features and bf16 precision issues in inputs_merger
        when running on TT/XLA devices."""
        inner = self._model.model

        with torch.no_grad():
            inputs_embeds = inner.text_model.get_input_embeddings()(input_ids)

            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pv = pixel_values.to(dtype=inner.dtype)
            pv = pv.view(batch_size * num_images, *pixel_values.shape[2:])

            if pixel_attention_mask is not None:
                pam = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
            else:
                pam = torch.ones(
                    size=[pv.shape[i] for i in (0, 2, 3)],
                    dtype=torch.bool,
                    device=pv.device,
                )

            patch_size = inner.config.vision_config.patch_size
            patches_subgrid = pam.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(
                dimension=2, size=patch_size, step=patch_size
            )
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            image_outputs = inner.vision_model(
                pixel_values=pv,
                patch_attention_mask=patch_attention_mask,
                return_dict=True,
            )
            image_hidden_states = inner.connector(image_outputs.last_hidden_state)

            inputs_embeds = inner.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        return inputs_embeds

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_path = get_file(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
        )
        from PIL import Image

        image = Image.open(str(image_path)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Can you describe this image?"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
            do_image_splitting=False,
        )

        inputs_embeds = self._precompute_merged_embeds(
            inputs["input_ids"],
            inputs["pixel_values"],
            inputs.get("pixel_attention_mask"),
        )
        attention_mask = inputs["attention_mask"]

        if dtype_override:
            inputs_embeds = cast_input_to_type(inputs_embeds, dtype_override)

        result = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

        if batch_size > 1:
            for key in result:
                if torch.is_tensor(result[key]):
                    result[key] = result[key].repeat_interleave(batch_size, dim=0)

        return result

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.processor is None:
            self._load_processor()

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.processor.decode(token_ids[0], skip_special_tokens=True)
