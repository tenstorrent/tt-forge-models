# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CV-VLM model loader implementation for image-text-to-text tasks (PyTorch).
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Optional

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
    """Available CV-VLM model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """CV-VLM model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="shixuanleong/cv-vlm",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CV-VLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        # transformers 5.x renamed additional_special_tokens to extra_special_tokens on
        # tokenizers; the remote processor code reads additional_special_tokens to extend
        # the token list.
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
            PreTrainedTokenizerBase.additional_special_tokens = property(
                lambda self: getattr(self, "extra_special_tokens", [])
            )

        # transformers 5.x changed CLIPImageProcessor to use_fast=True by default,
        # which produces non-square pixel_values. The DaViT encoder requires square
        # feature maps, so force the original slow processor.
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.x moved forced_bos_token_id out of PretrainedConfig; the remote
        # Florence2LanguageConfig accesses self.forced_bos_token_id after super().__init__()
        # which no longer sets it.
        from transformers import PretrainedConfig, PreTrainedModel
        if not hasattr(PretrainedConfig, "forced_bos_token_id"):
            PretrainedConfig.forced_bos_token_id = None

        # transformers 5.x always initializes models in a meta-device context; DaViT.__init__
        # calls .item() on a torch.linspace() result which is incompatible with meta tensors.
        # Temporarily remove the meta device context during model initialization.
        _orig_get_init_context = PreTrainedModel.get_init_context.__func__

        @classmethod
        def _get_init_context_no_meta(cls, dtype, is_quantized, _is_ds_init_called):
            contexts = _orig_get_init_context(cls, dtype, is_quantized, _is_ds_init_called)
            return [c for c in contexts if not (isinstance(c, torch.device) and c.type == "meta")]

        PreTrainedModel.get_init_context = _get_init_context_no_meta
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                attn_implementation="eager",
                **model_kwargs,
            )
        finally:
            PreTrainedModel.get_init_context = classmethod(_orig_get_init_context)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_path)).convert("RGB")

        prompt = "<CAPTION>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # Florence-2 based model requires decoder_input_ids
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
