# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Model loader implementation for GLM-4-Voice causal language modeling.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from typing import Optional
import torch

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
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
    get_static_cache_decode_inputs,
)


class ModelVariant(StrEnum):
    """Available model variants for GLM-4-Voice causal LM."""

    GLM_4_VOICE_9B = "glm-4-voice-9b"


class ModelLoader(ForgeModel):
    """Model loader implementation for GLM-4-Voice causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GLM_4_VOICE_9B: LLMModelConfig(
            pretrained_model_name="zai-org/glm-4-voice-9b",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GLM_4_VOICE_9B

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

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
        self.seq_len = None
        self.config = None
        self.num_layers = num_layers

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
            model="GLM-4-Voice",
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
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **tokenizer_kwargs
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        # The remote modeling_chatglm.py expects config.max_length but
        # ChatGLMConfig only provides seq_length.
        if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
            config.max_length = config.seq_length
        # transformers 5.x removed use_cache from PretrainedConfig defaults;
        # the remote forward() still reads self.config.use_cache.
        if not hasattr(config, "use_cache"):
            config.use_cache = True
        model_kwargs["config"] = config
        model_kwargs |= kwargs

        # The remote ChatGLMForConditionalGeneration.__init__ does not call
        # self.post_init(), which is required in transformers 5.x to initialize
        # all_tied_weights_keys before _finalize_model_loading accesses it.
        chatglm_cls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMForConditionalGeneration",
            pretrained_model_name,
        )
        if not getattr(chatglm_cls, "_post_init_patched", False):
            _orig_init = chatglm_cls.__init__

            def _patched_init(self, *args, **kw):
                _orig_init(self, *args, **kw)
                if not hasattr(self, "all_tied_weights_keys"):
                    self.post_init()

            chatglm_cls.__init__ = _patched_init
            chatglm_cls._post_init_patched = True

        chatglm_model_cls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMModel",
            pretrained_model_name,
        )
        if not getattr(chatglm_model_cls, "_get_masks_patched", False):
            # The remote get_masks() ends with full_attention_mask.unsqueeze_(1),
            # an in-place op on a bool tensor. XLA's xtensor accessor rejects
            # XLABoolType for in-place writes, causing BackendCompilerFailed.
            # Fully replace with a functionally identical version using out-of-place ops.
            def _patched_get_masks(self, input_ids, past_key_values, padding_mask=None):
                if self.config._attn_implementation == "flash_attention_2":
                    if padding_mask is not None and not padding_mask.all():
                        return padding_mask
                    return None
                batch_size, seq_length = input_ids.shape
                full_attention_mask = torch.ones(
                    batch_size, seq_length, seq_length, device=input_ids.device
                ).tril()
                past_length = 0
                if past_key_values:
                    past_length = past_key_values[0][0].shape[2]
                if past_length:
                    full_attention_mask = torch.cat(
                        (
                            torch.ones(
                                batch_size, seq_length, past_length, device=input_ids.device
                            ),
                            full_attention_mask,
                        ),
                        dim=-1,
                    )
                if padding_mask is not None:
                    full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
                if not past_length and padding_mask is not None:
                    full_attention_mask = full_attention_mask - (padding_mask.unsqueeze(-1) - 1)
                full_attention_mask = (full_attention_mask < 0.5).bool().unsqueeze(1)
                return full_attention_mask

            chatglm_model_cls.get_masks = _patched_get_masks
            chatglm_model_cls._get_masks_patched = True

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this instance's variant settings.

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

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        # position_ids from the tokenizer covers only the original (unpadded) tokens.
        # Passing it would cause rotary_pos_emb to be indexed for only those positions
        # while query/key have the full padded length, producing a shape mismatch.
        inputs.pop("position_ids", None)
        return inputs

    def load_inputs_decode(self, dtype_override=None, batch_size=1):
        """Load decode-step inputs (single token + static KV cache).
        Attention mask is intentionally omitted for single-batch decode. Defaults to steady-state decode.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        if self.config is None:
            self.load_config()

        max_cache_len = self._variant_config.max_length
        self.seq_len = 1

        return get_static_cache_decode_inputs(
            tokenizer=self.tokenizer,
            config=self.config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=dtype_override,
        )

    def decode_output(self, max_new_tokens, model, inputs, tokenizer):
        """Generates text.
        Args:
            max_new_tokens (int): The maximum number of new tokens to generate.
            model (torch.nn.Module): The language model used for token generation.
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len), representing tokenized text.
            tokenizer: The tokenizer used to decode token IDs into text.
        """
        current_pos = self.seq_len

        for _ in range(max_new_tokens):
            logits = model(*inputs)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            next_token_logits = logits[:, current_pos - 1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            inputs[0][:, current_pos] = next_token_id
            inputs[1][:, current_pos] = 1

            current_pos += 1

        valid_tokens = inputs[0][:, self.seq_len : current_pos].view(-1).tolist()
        answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)
        return answer

    def load_config(self):
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        return self.config
