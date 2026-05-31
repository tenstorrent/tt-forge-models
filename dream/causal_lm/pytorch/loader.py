# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dream model loader implementation for diffusion language modeling.

Dream (Dream-org/Dream-v0-Base-7B) is a 7B diffusion language model built on a
Qwen2-style transformer backbone. It is distributed as a custom-code model on
HuggingFace (``trust_remote_code=True``): the ``AutoModel`` auto-class maps to
``DreamModel``, whose forward pass returns per-token logits over the vocabulary
(``MaskedLMOutput``). The loader exercises a single bidirectional forward pass,
which is the building block of the model's iterative diffusion decoding.
"""

from transformers import AutoTokenizer, AutoModel, AutoConfig
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
from ....tools.utils import cast_input_to_type


def _ensure_default_rope_init():
    """Register a ``"default"`` RoPE initializer for Dream's custom modeling code.

    Dream's bundled ``modeling_dream.py`` (and its ``rope_scaling`` config) request
    ``ROPE_INIT_FUNCTIONS["default"]``, but transformers >= 5.x removed the
    ``"default"`` key (and the old ``_compute_default_rope_parameters`` helper),
    keeping only scaled variants. Re-register a standard, config-format-agnostic
    initializer so the model can build its rotary embeddings unchanged. No-op if a
    ``"default"`` entry already exists.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
        base = kwargs.get("base", None)
        if base is None:
            base = getattr(config, "rope_theta", None) or 10000.0
        partial_rotary_factor = getattr(config, "partial_rotary_factor", None) or 1.0
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
        attention_factor = 1.0  # default RoPE applies no post-scaling
        return inv_freq, attention_factor

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


class ModelVariant(StrEnum):
    """Available Dream model variants."""

    BASE_7B = "base_7b"


class ModelLoader(ForgeModel):
    """Dream diffusion language model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_7B: LLMModelConfig(
            pretrained_model_name="Dream-org/Dream-v0-Base-7B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_7B

    # Sample text for the diffusion LM forward pass
    sample_text = "The capital of France is"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None

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
            model="Dream",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Dream model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its default dtype.

        Returns:
            torch.nn.Module: The Dream model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Dream's custom modeling code relies on a "default" RoPE initializer that
        # newer transformers no longer ships; register a compatible one first.
        _ensure_default_rope_init()

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)

        # Dream defaults to use_cache=True, which builds a DynamicCache on every
        # forward. We only exercise a single (prefill) forward pass, so disable
        # the cache to keep the traced graph free of cache bookkeeping.
        model.config.use_cache = False

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Dream model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids) suitable for the diffusion LM forward pass.

        Note:
            Dream uses bidirectional (non-causal) attention. With an unpadded
            sequence, its own generation path treats the attention mask as
            "full" — i.e. passes ``None`` to ``scaled_dot_product_attention`` so
            every token attends to every other token. Passing a raw 2D integer
            mask instead trips SDPA's dtype check, so we omit the mask and let
            the model attend fully, matching Dream's intended behavior.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        tokenized = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].repeat_interleave(batch_size, dim=0)
        self.seq_len = input_ids.shape[-1]

        inputs = {"input_ids": input_ids}

        # Only convert dtype if explicitly requested (no-op for integer tensors)
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def load_config(self):
        """Load and return the configuration for the Dream model variant.

        Returns:
            The configuration object for the Dream model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
