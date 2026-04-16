# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
X-VLA vision-language-action model loader implementation (PyTorch).
"""

import torch
from transformers import AutoModel, AutoProcessor
from typing import Optional
from PIL import Image

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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available X-VLA model variants."""

    PT = "Pt"


class ModelLoader(ForgeModel):
    """X-VLA vision-language-action model loader implementation."""

    _VARIANTS = {
        ModelVariant.PT: ModelConfig(
            pretrained_model_name="2toINF/X-VLA-Pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="X-VLA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        # The tokenizer shipped by this model lacks special token configuration.
        # Set BART-standard special tokens (ids 0=<s>, 1=<pad>, 2=</s>).
        tok = self.processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = "<pad>"
            tok.pad_token_id = 1
        if tok.eos_token is None:
            tok.eos_token = "</s>"
            tok.eos_token_id = 2
        if tok.bos_token is None:
            tok.bos_token = "<s>"
            tok.bos_token_id = 0
        return self.processor

    @staticmethod
    def _patch_florence2_remote_code(pretrained_model_name):
        """Patch Florence2 remote code for transformers>=5.0 compatibility.

        Fixes two issues:
        1. Florence2LanguageConfig accesses self.forced_bos_token_id after
           super().__init__(), but transformers 5.x removed this default.
        2. Florence2PreTrainedModel._supports_sdpa is a property that accesses
           self.language_model, which doesn't exist during __init__. When the
           property raises AttributeError, the nn.Module.__getattr__ chain
           intercepts it. Fix by shadowing with a plain class attribute.
        """
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        lang_cfg = get_class_from_dynamic_module(
            "configuration_florence2.Florence2LanguageConfig",
            pretrained_model_name,
        )
        if not hasattr(lang_cfg, "forced_bos_token_id"):
            lang_cfg.forced_bos_token_id = None

        cond_gen_cls = get_class_from_dynamic_module(
            "modeling_florence2.Florence2ForConditionalGeneration",
            pretrained_model_name,
        )
        # Shadow the inherited property with a plain class attribute so
        # __init__ doesn't fail when self.language_model is not yet set.
        cond_gen_cls._supports_sdpa = True
        cond_gen_cls._supports_flash_attn_2 = False

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self._patch_florence2_remote_code(pretrained_model_name)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_path)).convert("RGB")

        prompt = "pick up the object"
        inputs = self.processor(language_instruction=prompt, images=[image])

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
