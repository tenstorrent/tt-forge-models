# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Emu3-Chat model loader implementation for multimodal visual question answering.
"""

import importlib.util
import os
import sys
import types
import torch
from PIL import Image
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
)
from huggingface_hub import snapshot_download

from ....tools.utils import get_file, cast_input_to_type
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
    """Available Emu3-Chat model variants."""

    EMU3_CHAT = "Chat"


class ModelLoader(ForgeModel):
    """Emu3-Chat model loader implementation for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.EMU3_CHAT: ModelConfig(
            pretrained_model_name="BAAI/Emu3-Chat",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EMU3_CHAT

    VISION_TOKENIZER_NAME = "BAAI/Emu3-VisionTokenizer"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None
        self.image_tokenizer = None
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Emu3",
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

    def _load_image_processor(self):
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.VISION_TOKENIZER_NAME, trust_remote_code=True
        )
        return self.image_processor

    def _load_image_tokenizer(self, dtype_override=None):
        kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override
        self.image_tokenizer = AutoModel.from_pretrained(
            self.VISION_TOKENIZER_NAME, **kwargs
        )
        self.image_tokenizer.eval()
        return self.image_tokenizer

    def _load_processor(self, dtype_override=None):
        if self.image_processor is None:
            self._load_image_processor()
        if self.image_tokenizer is None:
            self._load_image_tokenizer(dtype_override=dtype_override)
        if self.tokenizer is None:
            self._load_tokenizer()

        # Import the custom Emu3Processor from the model's remote code.
        # processing_emu3.py uses relative imports (from .utils_emu3 import ...),
        # so we must load it as part of a synthetic package.
        model_path = snapshot_download(
            self._variant_config.pretrained_model_name,
            allow_patterns=["processing_emu3.py", "utils_emu3.py"],
        )
        pkg_name = "_emu3_processing_pkg"
        parent_mod = types.ModuleType(pkg_name)
        parent_mod.__path__ = [model_path]
        parent_mod.__package__ = pkg_name
        sys.modules[pkg_name] = parent_mod

        utils_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.utils_emu3",
            os.path.join(model_path, "utils_emu3.py"),
        )
        utils_mod = importlib.util.module_from_spec(utils_spec)
        utils_mod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.utils_emu3"] = utils_mod
        utils_spec.loader.exec_module(utils_mod)

        proc_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.processing_emu3",
            os.path.join(model_path, "processing_emu3.py"),
        )
        proc_mod = importlib.util.module_from_spec(proc_spec)
        proc_mod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.processing_emu3"] = proc_mod
        proc_spec.loader.exec_module(proc_mod)

        Emu3Processor = proc_mod.Emu3Processor

        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        text = "What is shown in this image?"
        inputs = self.processor(
            text=text,
            image=image,
            mode="U",
            padding="longest",
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

    def decode_output(self, outputs, input_length=None):
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
