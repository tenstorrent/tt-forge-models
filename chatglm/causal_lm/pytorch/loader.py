# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChatGLM model loader implementation for causal language modeling.
"""
import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available ChatGLM model variants."""

    CHATGLM_6B = "6B"


class ModelLoader(ForgeModel):
    """ChatGLM model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.CHATGLM_6B: LLMModelConfig(
            pretrained_model_name="zai-org/chatglm-6b",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CHATGLM_6B

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ChatGLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _pre_patch_chatglm_tokenizer(pretrained_model_name):
        # Transformers 5.x calls get_vocab() during __init__ via _add_tokens(), which
        # accesses vocab_size -> sp_tokenizer before sp_tokenizer is initialized.
        # Pre-load the class via get_class_from_dynamic_module and patch __init__
        # to set sp_tokenizer first, before calling super().__init__().
        import sys

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module(
            "tokenization_chatglm.ChatGLMTokenizer",
            pretrained_model_name,
            trust_remote_code=True,
        )

        if getattr(cls, "_patched_sp_init", False):
            return

        mod = sys.modules.get(cls.__module__)
        if mod is None or not hasattr(mod, "SPTokenizer"):
            return

        SPTokenizer = mod.SPTokenizer
        orig_init = cls.__init__

        def patched_init(self, vocab_file, *args, num_image_tokens=20000, **kwargs):
            if not hasattr(self, "sp_tokenizer"):
                self.sp_tokenizer = SPTokenizer(
                    vocab_file, num_image_tokens=num_image_tokens
                )
            orig_init(
                self, vocab_file, *args, num_image_tokens=num_image_tokens, **kwargs
            )

        cls.__init__ = patched_init
        cls._patched_sp_init = True

        # Also patch _pad to accept the padding_side kwarg added in transformers 5.x.
        orig_pad = cls._pad

        def patched_pad(
            self,
            encoded_inputs,
            max_length=None,
            padding_strategy=None,
            pad_to_multiple_of=None,
            padding_side=None,
            return_attention_mask=None,
            **kwargs,
        ):
            old_ps = self.padding_side
            if padding_side is not None:
                self.padding_side = padding_side
            try:
                from transformers.utils import PaddingStrategy as _PS

                ps = (
                    padding_strategy if padding_strategy is not None else _PS.DO_NOT_PAD
                )
                return orig_pad(
                    self,
                    encoded_inputs,
                    max_length=max_length,
                    padding_strategy=ps,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_attention_mask=return_attention_mask,
                )
            finally:
                self.padding_side = old_ps

        cls._pad = patched_pad

    @staticmethod
    def _pre_patch_chatglm_model(pretrained_model_name):
        # ChatGLMForConditionalGeneration.__init__ doesn't call self.post_init(), which
        # transformers 5.x requires to set all_tied_weights_keys. Patch __init__ to
        # call post_init() after the original initialization completes.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMForConditionalGeneration",
            pretrained_model_name,
            trust_remote_code=True,
        )

        if getattr(cls, "_patched_post_init", False):
            return

        orig_init = cls.__init__

        def patched_init(self, config, *args, **kwargs):
            orig_init(self, config, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()

        cls.__init__ = patched_init
        cls._patched_post_init = True

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._pre_patch_chatglm_tokenizer(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **tokenizer_kwargs
        )

        # Set pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ChatGLM model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ChatGLM model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self._pre_patch_chatglm_model(pretrained_model_name)
        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        # ChatGLM hard-codes dtype=torch.half in __init__, which can leave some
        # parameters as float16 even when torch_dtype=bfloat16 is requested, causing
        # mixed-dtype errors in layer_norm.  Cast the whole model to ensure consistency.
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ChatGLM model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None, inputs=None):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            dtype_override: Optional torch.dtype to override the model's default dtype.
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded answer text
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        token_ids = torch.argmax(logits, dim=-1)
        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        return decoded[0] if len(decoded) == 1 else decoded
