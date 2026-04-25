# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Emu3-Chat model loader implementation for multimodal visual question answering.
"""

import torch
from PIL import Image
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

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

        Emu3ProcessorBase = get_class_from_dynamic_module(
            "processing_emu3.Emu3Processor",
            self._variant_config.pretrained_model_name,
        )

        # The model's Emu3Processor was written for older transformers; two fixes needed:
        # 1. ProcessorMixin.get_attributes() now introspects __init__ and "vision_tokenizer"
        #    matches "tokenizer" modality keyword, making it expect 3 components. Rename
        #    to "vq_model" to avoid modality matching.
        # 2. tokenizer.encode(list_of_strings) now returns [[id], [id], ...] instead of
        #    flat integers, so build_const_helper() fails in range(vis_start, vis_end+1).
        #    Override to use convert_tokens_to_ids() which returns flat integers.
        import sys as _sys
        from functools import partial as _partial
        from transformers.processing_utils import ProcessorMixin as _ProcessorMixin

        _proc_module = _sys.modules[Emu3ProcessorBase.__module__]
        _Emu3PrefixConstrainedLogitsHelper = (
            _proc_module.Emu3PrefixConstrainedLogitsHelper
        )

        class _FixedEmu3Processor(Emu3ProcessorBase):
            def __init__(
                self,
                image_processor=None,
                vq_model=None,
                tokenizer=None,
                chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:",
                prefix_template="{H}*{W}",
                visual_template=(
                    "<|visual token {token_id:0>6d}|>",
                    r"<\|visual token (\d+)\|>",
                ),
                **kwargs,
            ):
                assert vq_model is not None, "image tokenizer can not be None"
                self.vision_tokenizer = vq_model
                self.prefix_template = prefix_template
                self.visual_template = visual_template
                self.vis_tok_spatial_factor = 2 ** (
                    len(self.vision_tokenizer.config.ch_mult) - 1
                )
                _ProcessorMixin.__init__(
                    self, image_processor, tokenizer, chat_template=chat_template
                )
                self.const_helper = self.build_const_helper()

            def build_const_helper(self):
                (
                    img_token,
                    eoi_token,
                    eos_token,
                    eol_token,
                    eof_token,
                    pad_token,
                    vis_start,
                    vis_end,
                ) = self.tokenizer.convert_tokens_to_ids(
                    [
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
                )
                return _partial(
                    _Emu3PrefixConstrainedLogitsHelper,
                    img_token=img_token,
                    eoi_token=eoi_token,
                    eos_token=eos_token,
                    eol_token=eol_token,
                    eof_token=eof_token,
                    pad_token=pad_token,
                    visual_tokens=list(range(vis_start, vis_end + 1)),
                )

        self.processor = _FixedEmu3Processor(
            image_processor=self.image_processor,
            vq_model=self.image_tokenizer,
            tokenizer=self.tokenizer,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoConfig

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

        # The custom modeling_emu3.py expects rope_scaling["type"] but the model config
        # uses rope_scaling["rope_type"] (new transformers format). Patch the config so
        # the old custom code can find the "type" key, or clear it when type is "default".
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if config.rope_scaling is not None:
            rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
            if rope_type == "default":
                config.rope_scaling = None
            elif "type" not in config.rope_scaling:
                config.rope_scaling["type"] = rope_type
        model_kwargs["config"] = config

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
