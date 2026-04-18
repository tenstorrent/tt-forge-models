# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WeDLM model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

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
    """Available WeDLM model variants for causal language modeling."""

    WEDLM_8B_INSTRUCT = "8B_Instruct"


class ModelLoader(ForgeModel):
    """WeDLM model loader implementation for causal language modeling tasks."""

    @staticmethod
    def _patch_transformers_compat():
        """Patch transformers 5.x for WeDLM dynamic module compatibility."""
        from functools import wraps

        import transformers.utils.generic as generic_utils

        if not hasattr(generic_utils, "check_model_inputs"):

            def check_model_inputs(func=None, *, tie_last_hidden_states=True):
                def decorator(fn):
                    @wraps(fn)
                    def wrapper(self, *args, **kwargs):
                        return fn(self, *args, **kwargs)

                    return wrapper

                if func is not None:
                    return decorator(func)
                return decorator

            generic_utils.check_model_inputs = check_model_inputs

        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if "default" not in ROPE_INIT_FUNCTIONS:

            def _compute_default_rope_parameters(
                config=None, device=None, seq_len=None
            ):
                base = config.rope_theta
                partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or (
                    config.hidden_size // config.num_attention_heads
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

            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

        import transformers.modeling_utils as _mu

        _orig_init_weights = _mu.PreTrainedModel._init_weights

        def _patched_init_weights(self, module):
            if (
                "RotaryEmbedding" in module.__class__.__name__
                and hasattr(module, "_compute_default_rope_parameters")
                and not hasattr(module, "compute_default_rope_parameters")
            ):
                module.compute_default_rope_parameters = (
                    module._compute_default_rope_parameters
                )
            return _orig_init_weights(self, module)

        _mu.PreTrainedModel._init_weights = _patched_init_weights

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.WEDLM_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="tencent/WeDLM-8B-Instruct",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.WEDLM_8B_INSTRUCT

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="WeDLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the WeDLM model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The WeDLM model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        self._patch_transformers_compat()
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the WeDLM model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
