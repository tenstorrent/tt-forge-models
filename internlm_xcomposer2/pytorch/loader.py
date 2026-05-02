# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternLM-XComposer2 model loader implementation for multimodal visual question answering.
"""

import sys

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ...tools.utils import get_file
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


def _patch_clip_vision_tower_for_meta_ctx():
    """
    Patch CLIPVisionTower.__init__ to run outside the meta device context.

    transformers 5.x raises when from_pretrained is called inside the
    init_empty_weights() meta device context. CLIPVisionTower.__init__ calls
    load_model() (which calls CLIPVisionModel.from_pretrained) and resize_pos()
    (which creates new tensors/embeddings). Both must run outside meta context
    to load real CPU weights.

    This patch must be applied before AutoModelForCausalLM.from_pretrained;
    the caller must pre-load the remote model module so CLIPVisionTower is
    already in sys.modules before this is called.
    """
    for mod in sys.modules.values():
        if hasattr(mod, "CLIPVisionTower"):
            original_init = mod.CLIPVisionTower.__init__

            def _patched_init(self, *args, _orig=original_init, **kwargs):
                dev = torch.get_default_device()
                is_meta = dev is not None and dev.type == "meta"
                if is_meta:
                    torch.set_default_device(None)
                try:
                    _orig(self, *args, **kwargs)
                finally:
                    if is_meta:
                        torch.set_default_device("meta")

            mod.CLIPVisionTower.__init__ = _patched_init
            break


class ModelVariant(StrEnum):
    """Available InternLM-XComposer2 model variants."""

    INTERNLM_XCOMPOSER2_7B = "7B"
    INTERNLM_XCOMPOSER2_VL_7B = "VL_7B"


class ModelLoader(ForgeModel):
    """InternLM-XComposer2 model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.INTERNLM_XCOMPOSER2_7B: ModelConfig(
            pretrained_model_name="internlm/internlm-xcomposer2-7b",
        ),
        ModelVariant.INTERNLM_XCOMPOSER2_VL_7B: ModelConfig(
            pretrained_model_name="internlm/internlm-xcomposer2-vl-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERNLM_XCOMPOSER2_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="InternLM-XComposer2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "max_length"):
            config.max_length = 8192

        # Pre-load the remote model module so CLIPVisionTower is in sys.modules
        # before we patch it.  AutoModelForCausalLM.from_pretrained imports the
        # module INSIDE from_pretrained, which is too late to patch before
        # init_empty_weights() is entered.
        if hasattr(config, "auto_map") and "AutoModelForCausalLM" in config.auto_map:
            try:
                from transformers.dynamic_module_utils import (
                    get_class_from_dynamic_module,
                )

                get_class_from_dynamic_module(
                    config.auto_map["AutoModelForCausalLM"], pretrained_model_name
                )
            except Exception:
                pass

        # Patch CLIPVisionTower.__init__ to temporarily escape the meta device
        # context.  transformers 5.x raises when from_pretrained is called
        # inside init_empty_weights().
        _patch_clip_vision_tower_for_meta_ctx()

        model_kwargs = {
            "config": config,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        if self.model is None:
            raise RuntimeError("Model must be loaded before inputs via load_model()")

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")
        image = self.model.vis_processor(image)

        if dtype_override is not None:
            image = image.to(dtype_override)

        # Build query with image placeholder
        query = "<ImageHere>What is shown in this image?"

        # Tokenize the query
        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if image.dim() == 3:
            image = image.unsqueeze(0)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            image = image.repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": image,
            "use_cache": False,
        }

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.tokenizer is None:
            self._load_tokenizer()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)
