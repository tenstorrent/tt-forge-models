# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FlashVL-2B-Dynamic-ISS model loader implementation for multimodal visual question answering.
"""

import copy
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

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
    """Available FlashVL-2B-Dynamic-ISS model variants."""

    FLASHVL_2B_DYNAMIC_ISS = "2B_Dynamic_ISS"


class ModelLoader(ForgeModel):
    """FlashVL-2B-Dynamic-ISS model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.FLASHVL_2B_DYNAMIC_ISS: ModelConfig(
            pretrained_model_name="FlashVL/FlashVL-2B-Dynamic-ISS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLASHVL_2B_DYNAMIC_ISS

    default_query = "Describe this image."
    default_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FlashVL-2B-Dynamic-ISS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processors(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FlashVL-2B-Dynamic-ISS model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None or self.image_processor is None:
            self._load_processors()

        # FlashVLDynamicISSConfig.__init__ requires positional args, but transformers 5.x
        # calls self.__class__() with no args in to_diff_dict(). Setting has_no_defaults_at_init
        # tells transformers to skip that default-instance creation.
        config_cls = get_class_from_dynamic_module(
            "configuration_FlashVLDynamicISS.FlashVLDynamicISSConfig",
            pretrained_model_name,
        )
        config_cls.has_no_defaults_at_init = True

        # FlashVLDynamicISS was written for transformers 4.x and doesn't call self.post_init()
        # at the end of __init__. In transformers 5.x, post_init() sets all_tied_weights_keys
        # which is required by _finalize_model_loading -> _adjust_tied_keys_with_tied_pointers.
        model_cls = get_class_from_dynamic_module(
            "modeling_FlashVLDynamicISS.FlashVLDynamicISS",
            pretrained_model_name,
        )
        _original_init = model_cls.__init__

        def _patched_init(self, config):
            _original_init(self, config)
            self.post_init()

        model_cls.__init__ = _patched_init

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.tokenizer = self.tokenizer
        model.im_trans = self.image_processor
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for FlashVL-2B-Dynamic-ISS."""
        if self.tokenizer is None or self.image_processor is None:
            self._load_processors()

        image_file = get_file(self.default_image_url)
        image = Image.open(image_file).convert("RGB")

        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values

        # Tokenize with IMAGE_TOKEN_INDEX placeholders matching the model's expected format.
        # The model uses IMAGE_TOKEN_INDEX=-200 for the first token of each image and
        # IMAGE_PAD_TOKEN_INDEX=-201 for the remaining (image_token_num - 1) = 575 tokens.
        _IMAGE_TOKEN_INDEX = -200
        _IMAGE_PAD_TOKEN_INDEX = -201
        _IMAGE_TOKEN_NUM = 576

        tok = copy.deepcopy(self.tokenizer)
        tok.add_tokens(["<image>"], special_tokens=True)
        image_token_id = tok.convert_tokens_to_ids("<image>")
        tok.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

        def _apply_template(tok, messages, **kwargs):
            result = tok.apply_chat_template(messages, **kwargs)
            if hasattr(result, "input_ids"):
                ids = result.input_ids
                return ids.tolist() if hasattr(ids, "tolist") else list(ids)
            return list(result)

        system_ids = _apply_template(
            tok, [{"role": "system", "content": "You are a helpful assistant."}]
        )
        user_ids_raw = _apply_template(
            tok,
            [{"role": "user", "content": f"<image>\n{self.default_query}"}],
            add_generation_prompt=True,
        )

        expanded = []
        for tid in user_ids_raw:
            if tid == image_token_id:
                expanded.append(_IMAGE_TOKEN_INDEX)
                expanded.extend([_IMAGE_PAD_TOKEN_INDEX] * (_IMAGE_TOKEN_NUM - 1))
            else:
                expanded.append(tid)

        input_ids = torch.tensor([system_ids + expanded], dtype=torch.long)

        if batch_size > 1:
            input_ids = input_ids.repeat(batch_size, 1)
            pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"input_ids": input_ids, "pixel_values": pixel_values}
