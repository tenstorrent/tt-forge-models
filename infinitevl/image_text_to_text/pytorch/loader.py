# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InfiniteVL model loader implementation for image-text-to-text tasks.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
import transformers.modeling_utils as _tmu
from typing import Optional


def _default_rope_init_fn(config, device=None, **kwargs):
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _default_rope_init_fn


# Transformers 5.x changed _tied_weights_keys from list to dict format.
# Patch to handle models still using the old list format.
_orig_get_expanded_tied_weights_keys = PreTrainedModel.get_expanded_tied_weights_keys


def _patched_get_expanded_tied_weights_keys(self, all_submodels=False):
    if isinstance(self._tied_weights_keys, list):
        return {}
    return _orig_get_expanded_tied_weights_keys(self, all_submodels=all_submodels)


PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded_tied_weights_keys


# Transformers 5.x _init_weights for rope_type=="default" calls module.compute_default_rope_parameters
# instead of ROPE_INIT_FUNCTIONS["default"]. Patch _init_weights to look up ROPE_INIT_FUNCTIONS for all types.
_orig_init_weights = PreTrainedModel._init_weights


def _patched_init_weights(self, module):
    if (
        "RotaryEmbedding" in module.__class__.__name__
        and hasattr(module, "original_inv_freq")
        and getattr(module, "rope_type", None) == "default"
        and not hasattr(module, "compute_default_rope_parameters")
        and "default" in ROPE_INIT_FUNCTIONS
    ):
        buffer_value, _ = ROPE_INIT_FUNCTIONS["default"](module.config)
        _tmu.init.copy_(module.inv_freq, buffer_value)
        _tmu.init.copy_(module.original_inv_freq, buffer_value)
    else:
        _orig_init_weights(self, module)


PreTrainedModel._init_weights = _patched_init_weights

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available InfiniteVL model variants for image-text-to-text tasks."""

    INFINITEVL = "infinitevl"


class ModelLoader(ForgeModel):
    """InfiniteVL model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.INFINITEVL: LLMModelConfig(
            pretrained_model_name="hustvl/InfiniteVL",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INFINITEVL

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="InfiniteVL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
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

        model_kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if hasattr(config, "text_config") and not hasattr(
            config.text_config, "pad_token_id"
        ):
            config.text_config.pad_token_id = getattr(config, "eos_token_id", None)

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )
        # The model hardcodes _attn_implementation="flash_attention_2" in InfiniteVLSelfAttention.__init__,
        # overriding any attn_implementation kwarg. Reset it after loading for transformers 5.x compatibility.
        if hasattr(model.config, "text_config") and hasattr(
            model.config.text_config, "_attn_implementation"
        ):
            model.config.text_config._attn_implementation = "eager"
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
