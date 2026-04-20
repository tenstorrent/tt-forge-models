# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-V-4.5 model loader implementation for multimodal visual question answering.
"""

import torch
from copy import deepcopy
from transformers import AutoModel, AutoProcessor
from PIL import Image
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
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available MiniCPM-V-4.5 model variants."""

    MINICPM_V_4_5_INT4 = "V_4_5_int4"


class ModelLoader(ForgeModel):
    """MiniCPM-V-4.5 model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.MINICPM_V_4_5_INT4: ModelConfig(
            pretrained_model_name="openbmb/MiniCPM-V-4_5-int4",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINICPM_V_4_5_INT4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MiniCPM-V-4.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model

        return Wrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        question = "What is shown in this image?"

        msgs = [{"role": "user", "content": [image, question]}]

        images = []
        copy_msgs = deepcopy(msgs)
        for msg in copy_msgs:
            content = msg["content"]
            if isinstance(content, str):
                content = [content]
            cur_parts = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_parts.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_parts.append(c)
            msg["content"] = "\n".join(cur_parts)

        prompt = self.processor.tokenizer.apply_chat_template(
            copy_msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            [prompt], [images], return_tensors="pt", max_length=8192
        )

        inputs.pop("image_sizes", None)

        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        return dict(inputs)

    def decode_output(self, outputs, **kwargs):
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.processor.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if logits.dim() == 3:
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            else:
                next_token_id = torch.argmax(logits, dim=-1)
            return self.processor.tokenizer.decode(next_token_id)
