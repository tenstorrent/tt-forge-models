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


_CLIP_TOWER_NAME = "openai/clip-vit-large-patch14-336"


def _patch_clip_vision_tower_for_meta_init():
    """
    Patch CLIPVisionTower to work inside transformers 5.x meta device context.

    transformers 5.x initializes all models inside `torch.device("meta")`
    context, which patches nn.Module.register_parameter to move params to
    meta.  CLIPVisionTower.__init__ calls:
      1. load_model() → CLIPVisionModel.from_pretrained(...) which raises
         "You are using from_pretrained with a meta device context".
      2. resize_pos() → creates tensors that need real weights.

    Fix:
      - Patch load_model() to create the CLIP model structure from config
        only (no from_pretrained), letting the main VL checkpoint supply
        the actual weights when the outer from_pretrained materialises them.
      - Patch resize_pos() to be a no-op during meta init; the caller must
        invoke model.vit.resize_pos() AFTER the outer from_pretrained
        completes and the model has real weights.

    The caller must pre-load the remote module (so CLIPVisionTower is in
    sys.modules) before calling this function.  Returns the CLIPVisionTower
    class so the caller can find model.vit and call resize_pos() post-load.
    """
    from transformers import CLIPVisionModel

    import inspect

    for mod in list(sys.modules.values()):
        if not hasattr(mod, "CLIPVisionTower"):
            continue
        CLIPVisionTower = mod.CLIPVisionTower
        if not (inspect.isclass(CLIPVisionTower) and hasattr(CLIPVisionTower, "load_model")):
            continue

        original_load_model = CLIPVisionTower.load_model
        original_resize_pos = CLIPVisionTower.resize_pos

        def _struct_only_load_model(self):
            # Load only the config (no weights) so that the outer
            # from_pretrained can materialise the structure from the VL
            # checkpoint without ever calling from_pretrained inside the
            # meta context.
            clip_config = CLIPVisionModel.config_class.from_pretrained(
                _CLIP_TOWER_NAME
            )
            # resize_pos() resizes position embeddings from 24x24+1=577 to
            # 35x35+1=1226.  The VL checkpoint stores the already-resized
            # embeddings, so we must create the model with the right size
            # (image_size = 35 * patch_size = 490) to avoid a mismatch when
            # the checkpoint is loaded.  resize_pos() will see the embedding
            # is already the right size and skip the interpolation.
            clip_config.image_size = 35 * clip_config.patch_size
            self.vision_tower = CLIPVisionModel(clip_config)
            self.vision_tower.requires_grad_(False)
            self.is_loaded = True

        def _noop_resize_pos(self):
            # Skip position-embedding resize during meta init; called
            # manually by the loader after real weights are loaded.
            pass

        CLIPVisionTower.load_model = _struct_only_load_model
        CLIPVisionTower.resize_pos = _noop_resize_pos

        def _restore():
            CLIPVisionTower.load_model = original_load_model
            CLIPVisionTower.resize_pos = original_resize_pos

        return CLIPVisionTower, _restore

    return None, lambda: None


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
        # before we patch it.  AutoModelForCausalLM.from_pretrained imports
        # the module inside from_pretrained, which is too late to patch before
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

        # Patch CLIPVisionTower to work inside the meta device context.
        # load_model() will create the CLIP structure from config only; the
        # outer from_pretrained supplies the real weights.  resize_pos() is
        # deferred until after loading.
        clip_tower_cls, restore_patches = _patch_clip_vision_tower_for_meta_init()

        model_kwargs = {
            "config": config,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            restore_patches()

        # Apply position-embedding resize now that the VL checkpoint has
        # supplied real weights into the CLIP vision tower.
        if clip_tower_cls is not None and hasattr(model, "vit"):
            clip_tower_cls.resize_pos(model.vit)
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
