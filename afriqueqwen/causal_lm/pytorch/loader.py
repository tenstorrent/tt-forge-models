# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AfriqueQwen-14B-Fact (GGUF) model loader implementation for causal language modeling.

This loads the GGUF-quantized release ``mradermacher/AfriqueQwen-14B-Fact-full-i1-GGUF``,
a quantization of the Qwen3-14B based fine-tune ``israel/AfriqueQwen-14B-Fact-full``.
Transformers dequantizes the selected GGUF file into a full Qwen3 ``torch.nn.Module`` at
load time (the GGUF quantization is a storage format only; the materialized graph is a
standard dense Qwen3 model), which is what gets compiled for Tenstorrent hardware.
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
    """Available AfriqueQwen-14B-Fact GGUF variants for causal language modeling."""

    Q4_K_M = "14b_fact_i1_q4_k_m"


class ModelLoader(ForgeModel):
    """AfriqueQwen-14B-Fact (GGUF) loader for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/AfriqueQwen-14B-Fact-full-i1-GGUF",
            max_length=128,
        ),
    }

    # GGUF file (within the repo) to dequantize for each variant.
    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "AfriqueQwen-14B-Fact-full.i1-Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language model."

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
            model="AfriqueQwen 14B Fact (GGUF)",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        """Return the GGUF filename to dequantize for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from GGUF metadata.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AfriqueQwen-14B-Fact model instance for this variant.

        The selected GGUF file is dequantized into a dense Qwen3 model.

        ``transformers``' default ``from_pretrained(gguf_file=...)`` path holds the
        full float32 dequantized state dict (~56 GB for this 14B model) *and*
        allocates the target model on top, which OOMs a typical host. To keep peak
        host memory bounded we instead: (1) dequantize the GGUF tensors once,
        (2) downcast them in place to the requested dtype *before* allocating any
        model, then (3) build an empty model of that dtype and adopt the tensors
        with ``assign=True`` (no extra copy). Peak stays ~one model's worth.

        Args:
            dtype_override: Optional torch.dtype for the materialized weights.
                            Defaults to torch.float32 when not provided.

        Returns:
            torch.nn.Module: The Qwen3 model instance for causal language modeling.
        """
        from huggingface_hub import hf_hub_download
        from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._gguf_file()

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        target_dtype = dtype_override if dtype_override is not None else torch.float32

        # Config from GGUF metadata only (no tensors materialized here).
        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # A meta-device model (zero bytes) provides the gguf->HF tensor name map
        # that load_gguf_checkpoint requires, without allocating real weights.
        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(config)

        # Dequantize the GGUF tensors once (float32), then downcast in place to the
        # target dtype so the float32 buffers are released before model allocation.
        local_path = hf_hub_download(pretrained_model_name, gguf_file)
        ckpt = load_gguf_checkpoint(
            local_path, return_tensors=True, model_to_load=meta_model
        )
        del meta_model
        tensors = ckpt["tensors"]
        if target_dtype != torch.float32:
            for name in list(tensors.keys()):
                tensors[name] = tensors[name].to(target_dtype)

        # Build an empty model of the right dtype and adopt the tensors in place.
        # Skip the (expensive, ~minutes for 14B) random weight init since every
        # parameter is immediately overwritten by the dequantized GGUF tensors.
        from transformers.initialization import no_init_weights

        with no_init_weights():
            model = AutoModelForCausalLM.from_config(
                config, dtype=target_dtype, **kwargs
            )
        missing, unexpected = model.load_state_dict(
            tensors, strict=False, assign=True
        )
        del tensors, ckpt

        # lm_head may be tied / absent in the checkpoint; re-tie if needed.
        if missing:
            model.tie_weights()
        if unexpected:
            import warnings

            warnings.warn(f"Unexpected GGUF tensors ignored: {unexpected[:5]} ...")

        model = model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this variant's settings.

        Args:
            dtype_override: Unused for integer token inputs; kept for interface parity.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        # Prefer a chat template when the GGUF-derived tokenizer provides one,
        # otherwise fall back to plain tokenization of the sample text.
        prompt = self.sample_text
        if getattr(self.tokenizer, "chat_template", None):
            try:
                messages = [{"role": "user", "content": self.sample_text}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = self.sample_text

        inputs = self.tokenizer(
            [prompt],
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

    def load_config(self):
        """Load and return the configuration for the model variant from GGUF metadata.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )
        return self.config
