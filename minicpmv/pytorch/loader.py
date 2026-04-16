# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-V model loader implementation for multimodal visual question answering.
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
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
from ...tools.utils import cast_input_to_type, get_file


class MiniCPMVWrapper(torch.nn.Module):
    """Wrapper that adapts MiniCPMV's forward(data) signature to accept **kwargs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model(kwargs)


class ModelVariant(StrEnum):
    """Available MiniCPM-V model variants."""

    TINY_RANDOM_MINICPMV_2_6 = "Tiny_Random_MiniCPMV_2.6"


class ModelLoader(ForgeModel):
    """MiniCPM-V model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_MINICPMV_2_6: ModelConfig(
            pretrained_model_name="katuni4ka/tiny-random-minicpmv-2_6",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_MINICPMV_2_6

    sample_text = "What are these?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize MiniCPM-V model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MiniCPM-V",
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
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MiniCPM-V model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModel.from_pretrained(
            str(model_name),
            trust_remote_code=True,
            **kwargs,
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return MiniCPMVWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for MiniCPM-V."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
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

        try:
            text_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except ValueError:
            text_prompt = f"(<image>./</image>)\n{self.sample_text}"

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        if "position_ids" not in inputs and "input_ids" in inputs:
            seq_len = inputs["input_ids"].shape[-1]
            inputs["position_ids"] = torch.arange(seq_len).unsqueeze(0)

        if dtype_override:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

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
