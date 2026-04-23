# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Command R model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def _patch_command_r_gguf_support():
    """Patch transformers to add command-r GGUF architecture support.

    Transformers 5.x removed command-r from GGUF_CONFIG_MAPPING. We add it
    back and remap model_type from 'command-r' to 'cohere' after loading.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.configuration_utils as _config_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    import transformers.tokenization_utils_tokenizers as _tok_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    if "command-r" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("command-r")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["command-r"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "layer_norm_eps",
        "logit_scale": "logit_scale",
        "vocab_size": "vocab_size",
    }

    if "command-r" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["command-r"] = GGUFLlamaConverter

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load_gguf_checkpoint(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "command-r":
            result["config"]["model_type"] = "cohere"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_command_r_gguf_support()


class ModelVariant(StrEnum):
    """Available Command R model variants."""

    COMMAND_R7B_ARABIC = "7B_Arabic"
    COMMAND_R_PLUS = "Plus"
    COMMAND_R_PLUS_08_2024 = "Plus_08_2024"
    COMMAND_R_PLUS_GGUF = "Plus_GGUF"


class ModelLoader(ForgeModel):
    """Command R model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.COMMAND_R7B_ARABIC: LLMModelConfig(
            pretrained_model_name="CohereLabs/c4ai-command-r7b-arabic-02-2025",
        ),
        ModelVariant.COMMAND_R_PLUS: LLMModelConfig(
            pretrained_model_name="CohereLabs/c4ai-command-r-plus",
        ),
        ModelVariant.COMMAND_R_PLUS_08_2024: LLMModelConfig(
            pretrained_model_name="CohereLabs/c4ai-command-r-plus-08-2024",
        ),
        ModelVariant.COMMAND_R_PLUS_GGUF: LLMModelConfig(
            pretrained_model_name="pmysl/c4ai-command-r-plus-GGUF",
        ),
    }

    # GGUF files for quantized variants
    _GGUF_FILES = {
        ModelVariant.COMMAND_R_PLUS_GGUF: "command-r-plus-Q4_K_M-00001-of-00002.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.COMMAND_R7B_ARABIC

    # Shared configuration parameters
    sample_messages = [
        {"role": "user", "content": "ما هي عاصمة المملكة العربية السعودية؟"},
    ]

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
            model="Command R",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_gguf_variant(self):
        """Check if the current variant uses GGUF quantization."""
        return self._variant in self._GGUF_FILES

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

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
            tokenizer_kwargs["dtype"] = dtype_override

        # Pass gguf_file for GGUF variants
        if self._is_gguf_variant():
            tokenizer_kwargs["gguf_file"] = self._gguf_file

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Command R model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Command R model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        # Pass gguf_file for GGUF variants
        if self._is_gguf_variant():
            model_kwargs["gguf_file"] = self._gguf_file

        if self.num_layers is not None:
            from transformers import AutoConfig

            config_kwargs = {}
            if self._is_gguf_variant():
                config_kwargs["gguf_file"] = self._gguf_file
            config = AutoConfig.from_pretrained(pretrained_model_name, **config_kwargs)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Command R model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            self.sample_messages, tokenize=False
        )
        inputs = self.tokenizer(
            inputs, return_tensors="pt", return_token_type_ids=False
        )

        # Convert float32 tensors to the specified dtype if needed
        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
