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
from typing import Optional


def _default_rope_init(config, device=None, seq_len=None, layer_type=None):
    """Fallback for transformers 5.x which dropped the 'default' RoPE type."""
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device)
            / head_dim
        )
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

# transformers 5.x changed _tied_weights_keys from list to dict; models using the
# old list format crash in get_expanded_tied_weights_keys. Neutralise the list by
# treating it as "no programmatic tying" (weight tying still happens via tie_weights).
_orig_get_expanded = PreTrainedModel.get_expanded_tied_weights_keys


def _compat_get_expanded(self, all_submodels=False):
    if isinstance(self._tied_weights_keys, list):
        self._tied_weights_keys = None
    return _orig_get_expanded(self, all_submodels)


PreTrainedModel.get_expanded_tied_weights_keys = _compat_get_expanded

# transformers 5.x _init_weights tries module.compute_default_rope_parameters for
# rope_type=="default", but InfiniteVLRotaryEmbedding (and other old custom RoPE
# classes) don't have that method. Patch the base _init_weights to inject a fallback.
_orig_init_weights = PreTrainedModel._init_weights


def _compat_init_weights(self, module):
    if (
        "RotaryEmbedding" in type(module).__name__
        and getattr(module, "rope_type", None) == "default"
        and hasattr(module, "original_inv_freq")
        and not hasattr(module, "compute_default_rope_parameters")
    ):
        module.compute_default_rope_parameters = (
            lambda config, **kw: ROPE_INIT_FUNCTIONS["default"](config, **kw)
        )
    return _orig_init_weights(self, module)


PreTrainedModel._init_weights = _compat_init_weights

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

        # transformers 5.x removed pad_token_id from PretrainedConfig; patch it
        # explicitly so the model's embedding layer initialization doesn't fail.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config.text_config, "pad_token_id"):
            config.text_config.pad_token_id = None
        model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
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
