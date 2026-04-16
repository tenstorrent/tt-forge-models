# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-V-2_6-int4 model loader implementation for multimodal inference
"""

from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES
from PIL import Image

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import get_file


# Fix parallel styles issue for torch 2.7.0+ compatibility - works fine in torch 2.3.1
if ALL_PARALLEL_STYLES is None:
    import transformers.modeling_utils as mu

    mu.ALL_PARALLEL_STYLES = ["rowwise", "colwise", "headwise"]

# Monkey patch Resampler for compatibility - Fixes: Resampler doesn't have _initialize_weights method in torch 2.7.0
original_getattr = nn.Module.__getattr__


def patched_getattr(self, name):
    if name == "_initialize_weights" and self.__class__.__name__ == "Resampler":

        def _initialize_weights(module_self):
            if hasattr(module_self, "_init_weights"):
                module_self._init_weights(module_self)

        return _initialize_weights
    return original_getattr(self, name)


nn.Module.__getattr__ = patched_getattr


@dataclass
class MiniCPMVInt4Config(ModelConfig):
    """Configuration specific to MiniCPM-V-2_6-int4 models"""

    pretrained_model_name: str = "openbmb/MiniCPM-V-2_6-int4"


class ModelVariant(StrEnum):
    """Available MiniCPM-V-2_6-int4 model variants."""

    DEFAULT = "Default"


# Model variants configuration
_VARIANTS = {
    ModelVariant.DEFAULT: MiniCPMVInt4Config(
        pretrained_model_name="openbmb/MiniCPM-V-2_6-int4",
    ),
}


class MiniCPMVForwardWrapper(nn.Module):
    """Wrapper that translates keyword arguments into the data dict
    expected by MiniCPMV.forward(self, data, **kwargs)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values, tgt_sizes, image_bound, **kwargs):
        seq_len = input_ids.shape[1]
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )

        data = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": image_bound,
            "position_ids": position_ids,
        }

        return self.model(data, **kwargs)


class ModelLoader(ForgeModel):
    """MiniCPM-V-2_6-int4 model loader implementation for multimodal inference."""

    _VARIANTS = _VARIANTS
    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant=None):
        """Initialize MiniCPM-V-2_6-int4 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None) -> ModelInfo:
        """Get model information for MiniCPM-V-2_6-int4."""
        return ModelInfo(
            model="MiniCPM-V 2.6 int4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, **kwargs):
        """Load and return the MiniCPM-V-2_6-int4 model wrapped for forward compatibility."""
        config = self._variant_config

        model = AutoModel.from_pretrained(
            config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            **kwargs,
        )
        model.eval()

        if self.processor is None:
            self._load_processor()

        return MiniCPMVForwardWrapper(model)

    def load_inputs(self, **kwargs):
        """Load and return preprocessed tensor inputs for the model."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        question = "Describe this image in detail."

        messages = [
            {
                "role": "user",
                "content": [image, question],
            }
        ]

        # Build prompt text from messages
        copy_msgs = []
        images = []
        for msg in messages:
            content = msg["content"]
            cur_parts = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_parts.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_parts.append(c)
            copy_msgs.append({"role": msg["role"], "content": "\n".join(cur_parts)})

        prompt = self.processor.tokenizer.apply_chat_template(
            copy_msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            [prompt],
            [images],
            return_tensors="pt",
            max_length=8192,
        )

        # Remove image_sizes as the model doesn't use it in forward
        inputs.pop("image_sizes", None)
        inputs.pop("attention_mask", None)

        return dict(inputs)

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
