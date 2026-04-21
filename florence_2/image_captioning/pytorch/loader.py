# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Florence-2 image captioning model loader implementation (PyTorch).
"""

import sys

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import init, local_torch_dtype
from typing import Optional
from PIL import Image

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Florence-2 image captioning model variants."""

    BASE = "Base"
    BASE_FT = "Base_Ft"
    LARGE = "Large"
    SD3_CAPTIONER = "SD3-Captioner"


# Variants that use the <DESCRIPTION> prompt instead of <CAPTION>
_DESCRIPTION_VARIANTS = {ModelVariant.SD3_CAPTIONER}


def _patch_florence2_remote_code(pretrained_model_name):
    """Patch Florence-2 remote code for transformers 5.x compatibility.

    The remote code on HuggingFace accesses `self.forced_bos_token_id` before
    `PretrainedConfig.__init__` sets it, which fails in transformers 5.x where
    `__getattribute__` is stricter. Additionally, `get_init_context` uses
    `torch.device("meta")` unconditionally, which breaks the remote DaViT
    vision encoder that calls `.item()` on tensors during init.
    """
    config_cls = get_class_from_dynamic_module(
        "configuration_florence2.Florence2LanguageConfig",
        pretrained_model_name,
        trust_remote_code=True,
    )
    config_module = sys.modules[config_cls.__module__]

    original_lang_init = config_module.Florence2LanguageConfig.__init__

    def _patched_lang_config_init(self, *args, **kwargs):
        self.forced_bos_token_id = kwargs.get("forced_bos_token_id", None)
        original_lang_init(self, *args, **kwargs)

    config_module.Florence2LanguageConfig.__init__ = _patched_lang_config_init

    model_cls = get_class_from_dynamic_module(
        "modeling_florence2.Florence2ForConditionalGeneration",
        pretrained_model_name,
        trust_remote_code=True,
    )

    @classmethod
    def _no_meta_init_context(cls, dtype, is_quantized, _is_ds_init_called):
        return [local_torch_dtype(dtype, cls.__name__), init.no_init_weights()]

    model_cls.get_init_context = _no_meta_init_context


class ModelLoader(ForgeModel):
    """Florence-2 image captioning model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="microsoft/Florence-2-base",
        ),
        ModelVariant.BASE_FT: ModelConfig(
            pretrained_model_name="microsoft/Florence-2-base-ft",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="microsoft/Florence-2-large",
        ),
        ModelVariant.SD3_CAPTIONER: ModelConfig(
            pretrained_model_name="gokaygokay/Florence-2-SD3-Captioner",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Florence-2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_CAPT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor_components(self):
        name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.image_processor = CLIPImageProcessor.from_pretrained(name)

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        _patch_florence2_remote_code(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_processor_components()

        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_path)).convert("RGB")

        prompt = (
            "<DESCRIPTION>" if self._variant in _DESCRIPTION_VARIANTS else "<CAPTION>"
        )
        text_inputs = self.tokenizer(prompt, return_tensors="pt")
        pixel_values = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": pixel_values,
        }

        decoder_start_token_id = self.tokenizer.bos_token_id or 2
        inputs["decoder_input_ids"] = torch.full(
            (1, 1), decoder_start_token_id, dtype=torch.long
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

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

        if self.tokenizer is None:
            self._load_processor_components()

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
