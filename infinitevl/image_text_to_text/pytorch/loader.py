# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InfiniteVL model loader implementation for image-text-to-text tasks.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from typing import Optional


def _default_rope_init(config, device=None):
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

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

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        # transformers 5.x removed pad_token_id default from PretrainedConfig;
        # patch sub-config so model __init__ doesn't raise AttributeError
        if hasattr(config, "text_config") and not hasattr(
            config.text_config, "pad_token_id"
        ):
            config.text_config.pad_token_id = None

        # transformers 5.x changed _tied_weights_keys from list to dict;
        # preload and patch the model class before instantiation
        auto_map = getattr(config, "auto_map", {})
        causal_lm_ref = auto_map.get("AutoModelForCausalLM")
        if causal_lm_ref:
            try:
                model_cls = get_class_from_dynamic_module(
                    causal_lm_ref, pretrained_model_name
                )
                if isinstance(getattr(model_cls, "_tied_weights_keys", None), list):
                    model_cls._tied_weights_keys = {
                        "lm_head.weight": "model.language_model.embed_tokens.weight"
                    }
                # transformers 5.x _init_weights calls compute_default_rope_parameters
                # on any RotaryEmbedding with rope_type=="default"; patch the class
                import sys

                model_module = sys.modules.get(model_cls.__module__)
                if model_module is not None:
                    rotary_cls = getattr(
                        model_module, "InfiniteVLRotaryEmbedding", None
                    )
                    if rotary_cls is not None and not hasattr(
                        rotary_cls, "compute_default_rope_parameters"
                    ):

                        def _compute_default_rope_parameters(self, config=None):
                            cfg = config or self.config
                            head_dim = getattr(
                                cfg,
                                "head_dim",
                                cfg.hidden_size // cfg.num_attention_heads,
                            )
                            partial_rotary_factor = getattr(
                                cfg, "partial_rotary_factor", 1.0
                            )
                            dim = int(head_dim * partial_rotary_factor)
                            inv_freq = 1.0 / (
                                cfg.rope_theta
                                ** (
                                    torch.arange(0, dim, 2, dtype=torch.int64).float()
                                    / dim
                                )
                            )
                            return inv_freq, 1.0

                        rotary_cls.compute_default_rope_parameters = (
                            _compute_default_rope_parameters
                        )
            except Exception:
                pass

        model_kwargs = {
            "trust_remote_code": True,
            "config": config,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

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
