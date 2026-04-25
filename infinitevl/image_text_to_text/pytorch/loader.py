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


def _rope_default_init(config=None, device=None, seq_len=None, **rope_kwargs):
    """Compatibility shim: transformers 5.x removed 'default' from ROPE_INIT_FUNCTIONS."""
    if config is not None:
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
    else:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    # Do not move to device — meta-tensor contexts used by from_pretrained
    # fail if .to() is called on a meta tensor.
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _rope_default_init


# Transformers 5.x changed _tied_weights_keys from list to dict. Patch
# get_expanded_tied_weights_keys to handle old-style list format.
_orig_get_expanded = PreTrainedModel.get_expanded_tied_weights_keys


def _compat_get_expanded(self, all_submodels=False):
    if isinstance(self._tied_weights_keys, list):
        try:
            embed_module = self.get_input_embeddings()
            embed_path = next(
                f"{name}.weight"
                for name, mod in self.named_modules()
                if mod is embed_module
            )
            self._tied_weights_keys = {k: embed_path for k in self._tied_weights_keys}
        except (AttributeError, StopIteration):
            self._tied_weights_keys = {}
    return _orig_get_expanded(self, all_submodels=all_submodels)


PreTrainedModel.get_expanded_tied_weights_keys = _compat_get_expanded


# Transformers 5.x _init_weights calls module.compute_default_rope_parameters
# for rope_type='default', but custom RotaryEmbedding classes may not have it.
# Fall back to ROPE_INIT_FUNCTIONS["default"] in that case.
_orig_init_weights = PreTrainedModel._init_weights


@torch.no_grad()
def _compat_init_weights(self, module):
    if (
        "RotaryEmbedding" in module.__class__.__name__
        and hasattr(module, "original_inv_freq")
        and getattr(module, "rope_type", None) == "default"
        and not hasattr(module, "compute_default_rope_parameters")
    ):
        from transformers import initialization as init

        buffer_value, _ = ROPE_INIT_FUNCTIONS["default"](module.config)
        init.copy_(module.inv_freq, buffer_value)
        init.copy_(module.original_inv_freq, buffer_value)
        return
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

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Transformers 5.x no longer defaults sub-config attributes to None;
        # patch pad_token_id on text_config so the custom modeling code works.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if hasattr(config, "text_config"):
            config.text_config.__dict__.setdefault("pad_token_id", None)

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
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
