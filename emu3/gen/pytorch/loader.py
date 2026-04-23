# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Emu3-Gen model loader implementation for text-to-image generation.
"""

import importlib
from functools import partial
import torch
from typing import Optional
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ....tools.utils import cast_input_to_type
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
    """Available Emu3-Gen model variants."""

    EMU3_GEN = "Gen"


class ModelLoader(ForgeModel):
    """Emu3-Gen model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.EMU3_GEN: ModelConfig(
            pretrained_model_name="BAAI/Emu3-Gen",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EMU3_GEN

    VISION_TOKENIZER_NAME = "BAAI/Emu3-VisionTokenizer"

    sample_prompt = "a portrait of young girl. masterpiece, film grained, best quality."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None
        self.image_tokenizer = None
        self.processor = None
        self.model = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Emu3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            padding_side="left",
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

        _Emu3ProcessorBase = get_class_from_dynamic_module(
            "processing_emu3.Emu3Processor",
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

        # Emu3Processor remote code was written for older transformers where:
        # 1. super().__init__ accepted partial args (2 instead of all 3)
        # 2. tokenizer.encode([list]) returned flat [id, ...] not [[id], ...]
        # Both behaviours changed in transformers 5.x, so we patch via subclass.
        _base_module = importlib.import_module(_Emu3ProcessorBase.__module__)
        _PrefixHelper = _base_module.Emu3PrefixConstrainedLogitsHelper

        class Emu3Processor(_Emu3ProcessorBase):
            @classmethod
            def get_attributes(cls):
                return ["image_processor", "tokenizer"]

            def build_const_helper(self):
                tokens = [
                    self.tokenizer.img_token,
                    self.tokenizer.eoi_token,
                    self.tokenizer.eos_token,
                    self.tokenizer.eol_token,
                    self.tokenizer.eof_token,
                    self.tokenizer.pad_token,
                    self.visual_template[0].format(token_id=0),
                    self.visual_template[0].format(
                        token_id=self.vision_tokenizer.config.codebook_size - 1
                    ),
                ]
                ids = [self.tokenizer.encode(t)[0] for t in tokens]
                (
                    img_token,
                    eoi_token,
                    eos_token,
                    eol_token,
                    eof_token,
                    pad_token,
                    vis_start,
                    vis_end,
                ) = ids
                return partial(
                    _PrefixHelper,
                    img_token=img_token,
                    eoi_token=eoi_token,
                    eos_token=eos_token,
                    eol_token=eol_token,
                    eof_token=eof_token,
                    pad_token=pad_token,
                    visual_tokens=list(range(vis_start, vis_end + 1)),
                )

        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer
        )
        return self.processor

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        # Remote code uses old rope_scaling["type"] key; transformers 5.x emits
        # rope_type="default" meaning no scaling — clear it so _init_rope takes
        # the default (non-scaled) path.
        if (
            config.rope_scaling is not None
            and config.rope_scaling.get("rope_type") == "default"
        ):
            config.rope_scaling = None

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "config": config,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)
        if self.config is None:
            self.load_config()

        inputs = self.processor(
            text=self.sample_prompt,
            mode="G",
            ratio="1:1",
            image_area=self.config.image_area,
            return_tensors="pt",
            padding="longest",
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        result = dict(inputs)
        # image_size is used for constrained generation but not a forward() arg
        result.pop("image_size", None)
        # Remote code uses DynamicCache.get_usable_length which was removed in
        # transformers 5.x — disable caching to avoid the incompatibility.
        result["use_cache"] = False
        return result

    def decode_output(self, outputs):
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.processor.decode(outputs[0])

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.tokenizer.decode(next_token_id)
