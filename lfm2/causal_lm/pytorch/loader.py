# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LFM2 model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_grouped_mm_experts_forward():
    """Fix grouped_mm_experts_forward to use float histc input on non-CUDA devices.

    transformers uses expert_ids_g.int() for non-CPU devices, but torch.histc on
    CPU (where XLA ops fall back via partition_fx_graph_for_cpu_fallback) only
    supports float input. When device.type is "xla" the int path is taken, causing:
        NotImplementedError: "histogram_cpu" not implemented for 'Int'
    Fix: use float whenever device.type != "cuda".
    """
    import transformers.integrations.moe as moe_module

    if getattr(moe_module, "_lfm2_histc_patched", False):
        return

    _grouped_linear = moe_module._grouped_linear

    def _patched_gmef(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden_states.device
        num_top_k = top_k_index.size(-1)
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)

        token_idx = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1)
            .expand(-1, num_top_k)
            .reshape(-1)
        )
        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)

        selected_hidden_states = hidden_states[token_idx]

        perm = torch.argsort(expert_ids)
        inv_perm = torch.argsort(perm)
        expert_ids_g = expert_ids[perm]
        sample_weights_g = sample_weights[perm]
        selected_hidden_states_g = selected_hidden_states[perm]

        selected_gate_up = self.gate_up_proj
        selected_down = self.down_proj
        selected_gate_up_bias = self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
        selected_down_bias = self.down_proj_bias[expert_ids_g] if self.has_bias else None

        # CUDA supports int histc; CPU and XLA (which falls back to CPU) require float.
        histc_input = expert_ids_g.float() if device.type != "cuda" else expert_ids_g.int()
        num_tokens_per_expert = torch.histc(
            histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1
        )
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        gate_up_out = _grouped_linear(
            selected_hidden_states_g,
            selected_gate_up,
            selected_gate_up_bias,
            offsets,
            is_transposed=self.is_transposed,
        )
        gated_out = self._apply_gate(gate_up_out)
        out_per_sample_g = _grouped_linear(
            gated_out,
            selected_down,
            selected_down_bias,
            offsets,
            is_transposed=self.is_transposed,
        )
        out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)
        out_per_sample = out_per_sample_g[inv_perm]
        final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        return final_hidden_states.to(hidden_states.dtype)

    moe_module.grouped_mm_experts_forward = _patched_gmef
    moe_module.ExpertsInterface._global_mapping["grouped_mm"] = _patched_gmef
    moe_module._lfm2_histc_patched = True

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
    """Available LFM2 model variants for causal language modeling."""

    LFM2_24B_A2B = "lfm2_24b_a2b"
    LFM2_2_6B_MLX_4BIT = "lfm2_2_6b_mlx_4bit"
    LFM2_350M_MLX_8BIT = "lfm2_350m_mlx_8bit"
    LFM2_350M_UNSLOTH = "lfm2_350m_unsloth"


class ModelLoader(ForgeModel):
    """LFM2 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LFM2_24B_A2B: LLMModelConfig(
            pretrained_model_name="LiquidAI/LFM2-24B-A2B",
            max_length=128,
        ),
        ModelVariant.LFM2_2_6B_MLX_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/LFM2-2.6B-4bit",
            max_length=128,
        ),
        ModelVariant.LFM2_350M_MLX_8BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/LFM2-350M-8bit",
            max_length=128,
        ),
        ModelVariant.LFM2_350M_UNSLOTH: LLMModelConfig(
            pretrained_model_name="unsloth/LFM2-350M",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LFM2_24B_A2B

    # Shared configuration parameters
    sample_text = "The quick brown fox jumps over the lazy dog."

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
            model="LFM2",
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
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LFM2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The LFM2 model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        if self._variant in (
            ModelVariant.LFM2_2_6B_MLX_4BIT,
            ModelVariant.LFM2_350M_MLX_8BIT,
        ):
            model_kwargs["device_map"] = "cpu"
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        _patch_grouped_mm_experts_forward()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the LFM2 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {"role": "user", "content": self.sample_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the LFM2 model variant.

        Returns:
            The configuration object for the LFM2 model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
