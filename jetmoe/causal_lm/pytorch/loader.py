# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
JetMoE model loader implementation for causal language modeling.
"""
import torch
import torch.nn.functional as F
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


def _patched_jetmoe_moe_forward(self, layer_input):
    """Static per-expert masked matmul replacing expert_size.tolist() + split."""
    bsz, length, emb_size = layer_input.size()
    hidden = layer_input.reshape(-1, emb_size)
    N = hidden.size(0)

    logits = self.router.layer(hidden).float()
    top_k_logits, top_k_indices = logits.topk(self.router.top_k, dim=1)
    top_k_gates = torch.softmax(top_k_logits, dim=1).to(hidden.dtype)

    gate_matrix = hidden.new_zeros(N, self.router.num_experts)
    gate_matrix.scatter_(1, top_k_indices, top_k_gates)

    output = hidden.new_zeros(N, emb_size)
    for e in range(self.router.num_experts):
        gate_e = gate_matrix[:, e : e + 1]
        h = F.linear(hidden, self.input_linear.weight[e])
        h1, h2 = h.chunk(2, dim=-1)
        h = self.activation(h1) * h2
        h = F.linear(h, self.output_linear.weight[e])
        output = output + h * gate_e

    return output.view(bsz, length, emb_size) + self.bias


def _patched_jetmoa_map(self, layer_input):
    """Static per-expert Q-projection replacing expert_size.tolist() + split."""
    bsz, length, emb_size = layer_input.size()
    hidden = layer_input.reshape(-1, emb_size)
    N = hidden.size(0)

    logits = self.router.layer(hidden).float()
    top_k_logits, top_k_indices = logits.topk(self.router.top_k, dim=1)
    top_k_gates = torch.softmax(top_k_logits, dim=1).to(hidden.dtype)

    layer_output = hidden.new_zeros(N, self.top_k, self.hidden_size)
    for e in range(self.num_experts):
        mask_e = (top_k_indices == e).unsqueeze(-1).to(hidden.dtype)
        q_e = F.linear(hidden, self.input_linear.weight[e]).unsqueeze(1)
        layer_output = layer_output + q_e * mask_e

    layer_output = layer_output.view(bsz, length, self.top_k, self.hidden_size)
    topo_info = (top_k_indices, top_k_gates)
    return layer_output, logits, topo_info


def _patched_jetmoa_reduce(self, layer_input, topo_info):
    """Static per-expert O-projection replacing expert_size.tolist() + split."""
    top_k_indices, top_k_gates = topo_info
    bsz, length, k, hidden_size = layer_input.size()
    hidden = layer_input.reshape(-1, k, hidden_size)
    N = hidden.size(0)
    hidden_flat = hidden.reshape(-1, hidden_size)

    output = hidden.new_zeros(N, self.input_size)
    for e in range(self.num_experts):
        gate_e = (top_k_gates * (top_k_indices == e).to(top_k_gates.dtype))
        h_e = F.linear(hidden_flat, self.output_linear.weight[e]).view(N, k, self.input_size)
        output = output + (h_e * gate_e.unsqueeze(-1)).sum(dim=1)

    return output.view(bsz, length, self.input_size) + self.bias


def _patch_jetmoe_moe(model):
    from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoE, JetMoeMoA

    JetMoeMoE.forward = _patched_jetmoe_moe_forward
    JetMoeMoA.map = _patched_jetmoa_map
    JetMoeMoA.reduce = _patched_jetmoa_reduce


class ModelVariant(StrEnum):
    """Available JetMoE model variants for causal language modeling."""

    JETMOE_8B = "8B"


class ModelLoader(ForgeModel):
    """JetMoE model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.JETMOE_8B: LLMModelConfig(
            pretrained_model_name="jetmoe/jetmoe-8b",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.JETMOE_8B

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
            model="JetMoE",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the JetMoE model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The JetMoE model instance for causal language modeling.
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
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()
        self.config = model.config

        _patch_jetmoe_moe(model)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the JetMoE model with this instance's variant settings.

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

        inputs = self.tokenizer(
            [self.sample_text],
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
        """Load and return the configuration for the JetMoE model variant.

        Returns:
            The configuration object for the JetMoE model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        return self.config
