# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-V-2_6-int4 model loader implementation for multimodal inference
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
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


class MiniCPMVForwardWrapper(nn.Module):
    """Wraps MiniCPMV to accept processor output as kwargs instead of a 'data' dict."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values, tgt_sizes, image_bound, **kwargs):
        seq_len = input_ids.shape[1]
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        data = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": image_bound,
            "position_ids": position_ids,
        }
        return self.model.forward(data, **kwargs)


@dataclass
class MiniCPMVInt4Config(ModelConfig):
    """Configuration specific to MiniCPM-V-2_6-int4 models"""

    pretrained_model_name: str = "openbmb/MiniCPM-V-2_6-int4"


class ModelVariant(StrEnum):
    """Available MiniCPM-V-2_6-int4 model variants."""

    DEFAULT = "Default"


_VARIANTS = {
    ModelVariant.DEFAULT: MiniCPMVInt4Config(
        pretrained_model_name="openbmb/MiniCPM-V-2_6-int4",
    ),
}


class ModelLoader(ForgeModel):
    """MiniCPM-V-2_6-int4 model loader implementation for multimodal inference."""

    _VARIANTS = _VARIANTS
    DEFAULT_VARIANT = ModelVariant.DEFAULT

    sample_text = "Describe this image in detail."

    def __init__(self, variant=None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None) -> ModelInfo:
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

    def load_inputs(self, **kwargs) -> Dict[str, Any]:
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        msgs = [{"role": "user", "content": [image, self.sample_text]}]

        # Replicate the chat() method's input processing logic
        copy_msgs = deepcopy(msgs)
        images = []
        for msg in copy_msgs:
            content = msg["content"]
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        prompt = self.processor.tokenizer.apply_chat_template(
            copy_msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            [prompt], [images], return_tensors="pt", max_length=8192
        )
        inputs.pop("image_sizes", None)

        return dict(inputs)

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output
