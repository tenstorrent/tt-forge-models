# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 14B Instruct 1M (GGUF) Causal LM model loader implementation.

This loads the community GGUF quantization
``bartowski/Qwen2.5-14B-Instruct-1M-GGUF`` (base model:
``Qwen/Qwen2.5-14B-Instruct-1M``). HuggingFace ``transformers`` reads the
selected ``.gguf`` file, dequantizes the weights back to floating point and
materializes a standard ``Qwen2ForCausalLM`` module, so the downstream graph
compiled for Tenstorrent hardware is the same Qwen2 architecture as the base
model (with quantization-introduced weight noise).
"""

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM, AutoConfig
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
    """Available Qwen 2.5 14B Instruct 1M GGUF variants (by quantization)."""

    Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Qwen 2.5 14B Instruct 1M GGUF loader for causal language modeling tasks."""

    # The GGUF repository on HuggingFace. ``transformers`` downloads the single
    # ``gguf_file`` from this repo and dequantizes it on load.
    _GGUF_REPO = "bartowski/Qwen2.5-14B-Instruct-1M-GGUF"

    # Mapping from variant to the specific GGUF file within the repo.
    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "Qwen2.5-14B-Instruct-1M-Q4_K_M.gguf",
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.Q4_K_M: LLMModelConfig(
            pretrained_model_name=_GGUF_REPO,
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

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
            model="Qwen 2.5 14B Instruct 1M GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self) -> str:
        """The GGUF filename within the repo for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        The tokenizer is extracted directly from the GGUF file (the repo has no
        standalone tokenizer files).

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 2.5 14B Instruct 1M GGUF model instance.

        The 14B GGUF weights are dequantized to float32 by ``transformers``,
        producing a ~56 GB state dict for this model. Materializing a second
        full-precision (or even bfloat16) copy on top of that — which the
        standard ``from_pretrained(gguf_file=...)`` path does — exceeds host
        RAM. To keep the peak footprint near the size of a single dequantized
        copy, we:

          1. build a meta-device model purely to obtain the GGUF->HF tensor
             name mapping (zero bytes),
          2. dequantize the GGUF tensors into a state dict,
          3. cast that state dict to ``dtype_override`` in place (freeing the
             float32 storage as we go), and
          4. construct the real model directly in the target dtype and load the
             (now smaller) state dict into it.

        Args:
            dtype_override: Optional torch.dtype to cast the dequantized weights
                to (e.g. torch.bfloat16). If None, weights are kept as the
                float32 result of dequantization.

        Returns:
            torch.nn.Module: The Qwen2ForCausalLM model instance.
        """
        import gc

        from huggingface_hub import hf_hub_download
        from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        # Config is read from the GGUF metadata only (no tensors materialized).
        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file
        )

        # Local path to the GGUF file (downloaded + cached by huggingface_hub).
        gguf_path = hf_hub_download(
            repo_id=pretrained_model_name, filename=self._gguf_file
        )

        # 1. Meta model used solely for the GGUF -> HF tensor name mapping.
        with torch.device("meta"):
            meta_model = Qwen2ForCausalLM(config)

        # 2. Dequantize GGUF tensors into a (float32) state dict.
        checkpoint = load_gguf_checkpoint(
            gguf_path, return_tensors=True, model_to_load=meta_model
        )
        state_dict = checkpoint["tensors"]
        del meta_model
        gc.collect()

        # 3. Cast to the target dtype in place, freeing float32 storage.
        if dtype_override is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].to(dtype_override)
            gc.collect()

        # 4. Construct the real model directly in the target dtype and load.
        previous_dtype = torch.get_default_dtype()
        try:
            if dtype_override is not None:
                torch.set_default_dtype(dtype_override)
            model = Qwen2ForCausalLM(config, **kwargs)
        finally:
            torch.set_default_dtype(previous_dtype)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading GGUF weights: {unexpected[:5]} "
                f"({len(unexpected)} total)"
            )
        # Any "missing" keys are non-persistent buffers (e.g. rotary inv_freq)
        # that the freshly constructed model already initialized.
        del state_dict
        gc.collect()

        model.tie_weights()
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this variant's settings.

        Args:
            dtype_override: Unused for tokenized integer inputs; kept for API parity.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

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
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object for the Qwen 2.5 model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file,
        )
        return self.config
