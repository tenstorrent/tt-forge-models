# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MercuriusDream Huihui Qwen3.5-35B-A3B Abliterated MLX 2bit low model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    """Available MercuriusDream Huihui Qwen3.5-35B-A3B Abliterated MLX 2bit low model variants for causal language modeling."""

    MERCURIUSDREAM_HUIHUI_QWEN3_5_35B_A3B_ABLITERATED_MLX_2BIT_LOW = (
        "35B_A3B_Abliterated_MLX_2bit_low"
    )


class ModelLoader(ForgeModel):
    """MercuriusDream Huihui Qwen3.5-35B-A3B Abliterated MLX 2bit low model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MERCURIUSDREAM_HUIHUI_QWEN3_5_35B_A3B_ABLITERATED_MLX_2BIT_LOW: LLMModelConfig(
            pretrained_model_name="MercuriusDream/Huihui-Qwen3.5-35B-A3B-abliterated-MLX-2bit-low",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.MERCURIUSDREAM_HUIHUI_QWEN3_5_35B_A3B_ABLITERATED_MLX_2BIT_LOW
    )

    sample_text = "Give me a short introduction to large language model."

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
        return ModelInfo(
            model="MercuriusDream Huihui Qwen3.5-35B-A3B Abliterated MLX 2bit low",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

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
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        # Load config and extract text_config if this is a VLM wrapper (has text_config
        # nested inside, as in Qwen3.5-MoE conditional generation models). Also strip
        # MLX-format quantization_config which lacks the quant_method key required by
        # transformers and would cause a ValueError during loading.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "text_config"):
            config = config.text_config
        if (
            hasattr(config, "quantization_config")
            and isinstance(config.quantization_config, dict)
            and "quant_method" not in config.quantization_config
        ):
            del config.quantization_config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )
        model.eval()

        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
