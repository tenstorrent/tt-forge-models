# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.2 model loader implementation.

Uses a locally modified Transformer (model.py) instead of the original
HuggingFace model. The modifications are:
  1. Uses scipy.linalg.hadamard instead of fast_hadamard_transform (no CUDA required).
  2. Stubs out FP8 quantization (act_quant, fp8_gemm, fp8_index) that rely on
     custom tilelang kernels unsupported on TT hardware.
  3. Avoids torch.view_as_complex / view_as_real operations.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
    LLMModelConfig,
)


class ModelVariant(StrEnum):
    """Available DeepSeek V3.2 model variants."""

    DEEPSEEK_V3_2_EXP = "deepseek_v3_2_exp"


from .modified_model import ModelArgs, Transformer


@dataclass
class _CausalLMOutput:
    """Minimal output container satisfying ``output.logits`` expected by the benchmark."""

    logits: torch.Tensor


class DeepSeekV32ForCausalLM(nn.Module):
    """HuggingFace-compatible wrapper around the custom Transformer.

    The benchmark calls ``model(input_ids=..., past_key_values=...,
    cache_position=..., use_cache=...)`` and reads ``output.logits``.
    The custom Transformer uses ``forward(tokens, start_pos)`` and returns a
    raw ``[batch, vocab_size]`` tensor.  This wrapper bridges that gap without
    touching ``modified_model.py``.

    ``cache_position`` is ``[seq_len]`` on the prefill step and ``[1]`` on
    each decode step; its first value is the absolute position of the first
    token being processed, which maps directly to ``start_pos`` in the custom
    model's rotary-embedding indexing.

    The logits are unsqueezed from ``[batch, vocab]`` to ``[batch, 1, vocab]``
    so that ``logits[:, -1].argmax(dim=-1)`` in the decode loop works correctly.
    """

    def __init__(self, transformer: Transformer, config: PretrainedConfig):
        super().__init__()
        self.transformer = transformer
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> _CausalLMOutput:
        start_pos = int(cache_position[0].item()) if cache_position is not None else 0
        logits = self.transformer(tokens=input_ids, start_pos=start_pos)
        return _CausalLMOutput(logits=logits.unsqueeze(1))


class ModelLoader(ForgeModel):
    """DeepSeek V3.2 model loader using the locally modified Transformer."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_V3_2_EXP: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-V3.2-Exp",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_V3_2_EXP

    def __init__(
        self,
        variant=None,
        num_layers: Optional[int] = None,
        max_batch_size: int = 32,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional variant string. Unused; kept for API compatibility.
            num_layers: Number of transformer layers to instantiate.
                        If None, uses the ModelArgs default (27).
            max_batch_size: Maximum batch size for KV-cache allocation.
                            Must be >= the batch size used at inference time.
                            Defaults to 32 to match the benchmark default.
        """
        super().__init__(variant)
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.tokenizer = None
        self.model = None
        # self.config = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Return model metadata for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V3.2",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    # def _load_config(self, dtype_override=None):
    #     """Load the config for the current variant.

    #     Args:
    #         dtype_override: Optional torch.dtype to override the config's default dtype.

    #     Returns:
    #         The loaded config instance
    #     """
    #     # Get the pretrained model name from the instance's variant config
    #     pretrained_model_name = self._variant_config.pretrained_model_name

    #     # Load the config
    #     self.config = AutoConfig.from_pretrained(pretrained_model_name)

    #     return self.config

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the modified DeepSeek V3.2 Transformer.

        The model is constructed from ModelArgs defaults, overriding n_layers
        with the value passed at construction time.

        Args:
            dtype_override: Optional torch.dtype to cast the model to after
                            construction (e.g. torch.bfloat16).

        Returns:
            torch.nn.Module: The modified DeepSeek V3.2 Transformer in eval mode.
        """

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # if self.config is None:
        #     self._load_config(dtype_override=dtype_override)

        model_args_kwargs = {k: v for k, v in kwargs.items()}
        if self.num_layers is not None:
            model_args_kwargs["n_layers"] = self.num_layers
        model_args_kwargs.setdefault("max_batch_size", self.max_batch_size)
        args = ModelArgs(**model_args_kwargs)

        transformer = Transformer(args)

        if dtype_override is not None:
            transformer = transformer.to(dtype_override)

        transformer = transformer.eval()
        self._args = args

        config = PretrainedConfig(
            num_hidden_layers=args.n_layers,
            num_attention_heads=args.n_heads,
            num_key_value_heads=args.n_heads,
            hidden_size=args.dim,
            head_dim=args.v_head_dim,
        )

        return DeepSeekV32ForCausalLM(transformer, config)

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallelism.

        Args:
            num_devices: Number of devices available at runtime.

        Returns:
            Tuple of (mesh_shape, axis_names).
        """
        if num_devices == 32:
            return (4, 8), ("batch", "model")
        raise ValueError(
            f"DeepSeek V3.2 is only supported on Galaxy (32 devices), got {num_devices}"
        )

    def load_inputs(self, batch_size: int = 1, seq_len: int = 32):
        """Return sample token inputs for the model.

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Length of each input sequence.

        Returns:
            torch.Tensor: Integer token tensor of shape (batch_size, seq_len).
        """
        if not hasattr(self, "_args"):
            self.load_model()

        tokens = torch.randint(0, self._args.vocab_size, (batch_size, seq_len))
        return tokens
