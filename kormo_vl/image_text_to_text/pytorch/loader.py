# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KORMo-VL model loader implementation for image-text-to-text generation.
"""

from typing import Optional

from PIL import Image
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available KORMo-VL model variants."""

    KORMO_VL = "KORMo-VL/KORMo-VL"


class ModelLoader(ForgeModel):
    """KORMo-VL model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.KORMO_VL: ModelConfig(
            pretrained_model_name=str(ModelVariant.KORMO_VL),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KORMO_VL

    sample_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="KORMo-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        # use_fast=False: transformers 5.x switched LlavaOnevisionImageProcessor to
        # fast mode by default. The fast processor produces slightly different image
        # token counts than the slow one, causing token/feature count mismatches.
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        return self.processor

    @staticmethod
    def _register_kormo_classes():
        # KORMo-VL/KORMo-VL uses the built-in llava_onevision architecture with a
        # custom "kormo" text model whose code lives in KORMo-Team/KORMo-10B-sft.
        # LlavaOnevisionConfig.__init__ looks up CONFIG_MAPPING["kormo"] directly
        # before any auto_map resolution, so the class must be registered first.
        kormo_base_repo = "KORMo-Team/KORMo-10B-sft"
        KORMoConfig = get_class_from_dynamic_module(
            "_configuration_kormo.KORMoConfig", kormo_base_repo
        )
        KORMoModel = get_class_from_dynamic_module(
            "_modeling_kormo.KORMoModel", kormo_base_repo
        )
        KORMoForCausalLM = get_class_from_dynamic_module(
            "_modeling_kormo.KORMoForCausalLM", kormo_base_repo
        )
        AutoConfig.register("kormo", KORMoConfig, exist_ok=True)
        AutoModel.register(KORMoConfig, KORMoModel, exist_ok=True)
        AutoModelForCausalLM.register(KORMoConfig, KORMoForCausalLM, exist_ok=True)

    @staticmethod
    def _patch_llava_placeholder_mask():
        # LlavaOnevisionModel.get_placeholder_mask calls torch_compilable_check with
        # inputs_embeds[special_image_mask].numel() where special_image_mask is a 3D
        # bool tensor [batch, seq_len, hidden_size]. The boolean gather forces a
        # dynamic-size tensor onto TT device, overflowing L1 (96 MB CB vs 1.5 MB).
        # Replace with equivalent arithmetic using n_image_tokens (computed on CPU
        # from input_ids before the expand) times the static hidden_size dimension.
        import torch
        from transformers.models.llava_onevision.modeling_llava_onevision import (
            LlavaOnevisionModel,
        )
        from transformers.utils import torch_compilable_check

        def patched_get_placeholder_mask(
            self, input_ids, inputs_embeds, image_features=None, video_features=None
        ):
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

            # Compute token counts on CPU to avoid TT device int64 comparison bug
            # (TT gives sum=2928 instead of 2929 for input_ids==125041 on device).
            if input_ids is not None:
                n_image_tokens = int(
                    (input_ids.cpu() == self.config.image_token_id).sum()
                )
                n_video_tokens = int(
                    (special_video_mask.cpu()).sum()
                )
            else:
                n_image_tokens = int(special_image_mask.sum())
                n_video_tokens = int(special_video_mask.sum())
            special_image_mask = (
                special_image_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            if image_features is not None:
                torch_compilable_check(
                    n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                    f"Image features and image tokens do not match, "
                    f"tokens: {n_image_tokens}, features: {image_features.shape[0]}",
                )

            special_video_mask = (
                special_video_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            if video_features is not None:
                torch_compilable_check(
                    n_video_tokens * inputs_embeds.shape[-1] == video_features.numel(),
                    f"Video features and video tokens do not match, "
                    f"tokens: {n_video_tokens}, features: {video_features.shape[0]}",
                )
            return special_image_mask, special_video_mask

        LlavaOnevisionModel.get_placeholder_mask = patched_get_placeholder_mask

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the KORMo-VL model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        self._register_kormo_classes()
        self._patch_llava_placeholder_mask()

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for KORMo-VL."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
