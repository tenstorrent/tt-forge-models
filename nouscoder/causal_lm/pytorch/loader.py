# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NousCoder model loader implementation for causal language modeling.

NousResearch/NousCoder-14B is a Qwen3-architecture 14B coding model. This loader
consumes the GGUF-quantized release at bartowski/NousResearch_NousCoder-14B-GGUF;
transformers dequantizes the GGUF checkpoint into a standard Qwen3ForCausalLM at
load time (requires the `gguf` package).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from huggingface_hub import hf_hub_download
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
    """Available NousCoder model variants for causal language modeling."""

    NOUSCODER_14B = "14b"


class ModelLoader(ForgeModel):
    """NousCoder model loader implementation for causal language modeling tasks."""

    # GGUF repo holds only *.gguf files; transformers dequantizes the chosen
    # quant into a normal Qwen3 model. Q4_K_M is a K-quant that transformers
    # can fully dequantize (IQ-quants are not guaranteed to be supported).
    _GGUF_FILE = "NousResearch_NousCoder-14B-Q4_K_M.gguf"

    # GGUF dequantization expands every tensor to fp32; for a 14B model that is
    # ~56 GB resident, which OOMs the host under the test runner. Default to
    # bfloat16 (~28 GB) so the dequantized model fits; the device path also runs
    # in bfloat16, so this matches the comparison precision.
    _DEFAULT_DTYPE = torch.bfloat16

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.NOUSCODER_14B: LLMModelConfig(
            pretrained_model_name="bartowski/NousResearch_NousCoder-14B-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.NOUSCODER_14B

    # Shared configuration parameters
    sample_text = "Write a Python function that returns the nth Fibonacci number."

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
        self.config = None
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
            model="NousCoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILE,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NousCoder model instance for this instance's variant.

        The GGUF checkpoint is dequantized to a standard Qwen3 model. A direct
        ``AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`` call holds the
        full fp32 dequantized state dict (~56 GB for 14B) at the same time as it
        materializes the model, which OOMs the host. To stay within budget we
        replicate that path manually but cast the dequantized weights down to the
        target dtype (bfloat16 by default) *before* building the model, so the
        fp32 state dict and the model never coexist at full size.

        Args:
            dtype_override: Optional torch.dtype for the model weights. Defaults
                            to bfloat16, which also matches the device precision.

        Returns:
            torch.nn.Module: The NousCoder model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else self._DEFAULT_DTYPE

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self._GGUF_FILE
        )
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        # A meta-device dummy model is only needed to build the GGUF->HF weight
        # name map used by load_gguf_checkpoint (mirrors transformers internals).
        gguf_path = hf_hub_download(pretrained_model_name, self._GGUF_FILE)
        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(config)
        state_dict = load_gguf_checkpoint(
            gguf_path, return_tensors=True, model_to_load=dummy_model
        )["tensors"]

        # Cast the (fp32) dequantized tensors in place to free fp32 memory before
        # the model is materialized.
        for name in list(state_dict.keys()):
            state_dict[name] = state_dict[name].to(dtype)

        # Build the real model on CPU (initializes non-persistent buffers such as
        # rotary inv_freq) and assign the dequantized weights without copying.
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, **kwargs)
        model.load_state_dict(state_dict, strict=False, assign=True)
        del state_dict
        model = model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the NousCoder model with this instance's variant settings.

        Args:
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        # Use chat template (Qwen3 tokenizer)
        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the NousCoder model variant.

        Returns:
            The configuration object for the NousCoder model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILE,
        )

        return self.config
