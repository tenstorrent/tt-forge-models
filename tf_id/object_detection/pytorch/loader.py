# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TF-ID (Table/Figure IDentifier) object detection model loader implementation (PyTorch).
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module
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


def _patch_florence2_remote_code(repo_id):
    """Patch custom Florence2 classes from HuggingFace hub for transformers 5.x compat."""
    lang_cfg = get_class_from_dynamic_module(
        "configuration_florence2.Florence2LanguageConfig", repo_id
    )
    lang_cfg.forced_bos_token_id = None

    davit_cls = get_class_from_dynamic_module("modeling_florence2.DaViT", repo_id)
    if not getattr(davit_cls, "_meta_patched", False):
        _orig_init = davit_cls.__init__

        def _patched_init(self, *args, _orig=_orig_init, **kwargs):
            orig_item = torch.Tensor.item

            def safe_item(tensor):
                if tensor.is_meta:
                    return 0.0
                return orig_item(tensor)

            torch.Tensor.item = safe_item
            try:
                _orig(self, *args, **kwargs)
            finally:
                torch.Tensor.item = orig_item

        davit_cls.__init__ = _patched_init
        davit_cls._meta_patched = True

    processor_cls = get_class_from_dynamic_module(
        "processing_florence2.Florence2Processor", repo_id
    )
    if not getattr(processor_cls, "_tok_patched", False):
        _orig_proc_init = processor_cls.__init__

        def _patched_proc_init(self, *args, _orig=_orig_proc_init, **kwargs):
            for arg in list(args) + list(kwargs.values()):
                if hasattr(arg, "add_special_tokens") and not hasattr(
                    arg, "additional_special_tokens"
                ):
                    arg.additional_special_tokens = []
            _orig(self, *args, **kwargs)

        processor_cls.__init__ = _patched_proc_init
        processor_cls._tok_patched = True


class ModelVariant(StrEnum):
    """Available TF-ID model variants for object detection."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """TF-ID model loader implementation for table/figure detection tasks."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="yifeihu/TF-ID-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TF-ID",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        _patch_florence2_remote_code(self._variant_config.pretrained_model_name)
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        return self.processor

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
        if self.processor is None:
            self._load_processor()

        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_path)).convert("RGB")

        prompt = "<OD>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # TF-ID is a Florence-2 based seq2seq model that requires decoder_input_ids
        decoder_start_token_id = self.processor.tokenizer.bos_token_id or 2
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
