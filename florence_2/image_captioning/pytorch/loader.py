# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Florence-2 image captioning model loader implementation (PyTorch).
"""

import torch
from transformers import (
    Florence2ForConditionalGeneration,
    Florence2Config,
)
from transformers.models.bart.configuration_bart import BartConfig
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available Florence-2 image captioning model variants."""

    BASE = "Base"
    BASE_FT = "Base_Ft"
    LARGE = "Large"
    SD3_CAPTIONER = "SD3-Captioner"


_DESCRIPTION_VARIANTS = {ModelVariant.SD3_CAPTIONER}


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

    # Number of image feature tokens produced by the vision encoder for 768x768 input
    _NUM_IMAGE_TOKENS = 577

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

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

    def load_model(self, *, dtype_override=None, **kwargs):
        text_config = BartConfig(vocab_size=51290)
        config = Florence2Config(text_config=text_config.to_dict())
        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model = Florence2ForConditionalGeneration(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        image_token_id = 51289
        bos_id = 0
        eos_id = 2

        input_ids = torch.cat(
            [
                torch.tensor([[bos_id]]),
                torch.full((1, self._NUM_IMAGE_TOKENS), image_token_id),
                torch.tensor([[eos_id]]),
            ],
            dim=1,
        )
        pixel_values = torch.randn(1, 3, 768, 768)
        decoder_input_ids = torch.tensor([[eos_id]], dtype=torch.long)

        inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
        }

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

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return str(token_ids[0].tolist())
