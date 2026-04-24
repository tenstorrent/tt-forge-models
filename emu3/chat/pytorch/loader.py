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

        # Download all remote Python files to support relative imports
        model_path = snapshot_download(
            self._variant_config.pretrained_model_name,
            allow_patterns=["*.py"],
        )

        # Load processing_emu3 as part of a synthetic package so relative
        # imports (e.g. `from .utils_emu3 import …`) resolve correctly.
        pkg_name = "emu3_remote_code"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [model_path]
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg

        def _load_submodule(name):
            full_name = f"{pkg_name}.{name}"
            if full_name not in sys.modules:
                path = os.path.join(model_path, f"{name}.py")
                spec = importlib.util.spec_from_file_location(full_name, path)
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = pkg_name
                sys.modules[full_name] = mod
                spec.loader.exec_module(mod)

        utils_path = os.path.join(model_path, "utils_emu3.py")
        if os.path.exists(utils_path):
            _load_submodule("utils_emu3")
        _load_submodule("processing_emu3")

        _Emu3Processor = sys.modules[f"{pkg_name}.processing_emu3"].Emu3Processor

        # Newer transformers ProcessorMixin.__init__ infers required args from the
        # __init__ signature and validates count/types strictly.  The upstream
        # Emu3Processor.__init__ was written for an older API and only forwards 2
        # of the 3 required positional args to super().__init__, causing a
        # ValueError.  Subclass it to bypass the broken super().__init__ call.
        class _PatchedEmu3Processor(_Emu3Processor):
            def __init__(
                self,
                image_processor=None,
                vision_tokenizer=None,
                tokenizer=None,
                **kwargs,
            ):
                assert vision_tokenizer is not None, "image tokenizer can not be None"
                self.vision_tokenizer = vision_tokenizer
                self.prefix_template = kwargs.pop("prefix_template", "{H}*{W}")
                self.visual_template = kwargs.pop(
                    "visual_template",
                    (
                        "<|visual token {token_id:0>6d}|>",
                        r"<\|visual token (\d+)\|>",
                    ),
                )
                self.vis_tok_spatial_factor = 2 ** (
                    len(vision_tokenizer.config.ch_mult) - 1
                )
                # Set attrs that ProcessorMixin.__init__ would normally set
                self.chat_template = kwargs.pop(
                    "chat_template",
                    "You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:",
                )
                self.image_processor = image_processor
                self.tokenizer = tokenizer
                # const_helper is only used in generation mode ("G"), not
                # inference/understanding mode ("U"), so skip build_const_helper
                # to avoid tokenizer API incompatibility with newer transformers.
                self.const_helper = None

        self.processor = _PatchedEmu3Processor(
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
