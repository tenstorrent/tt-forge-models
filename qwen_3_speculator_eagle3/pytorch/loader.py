# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 8B EAGLE3 speculator model loader implementation for speculative decoding.
"""

import types

import torch
from speculators import SpeculatorModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _eagle3_forward(
    self,
    input_ids,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    """Forward pass for EAGLE-3 speculator (speculators 0.2.0 has a stub)."""
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    inputs_embeds = self.embed_tokens(input_ids)
    fused_hidden = self.fc(hidden_states)
    layer_input = torch.cat([inputs_embeds, fused_hidden], dim=-1)

    batch_size, seq_length = layer_input.shape[:2]
    if attention_mask is not None and attention_mask.dim() == 2:
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask,
        )

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values else 0
        )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

    if position_ids is None:
        device = hidden_states.device
        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

    layer_outputs = self.layers[0](
        layer_input,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_values[0] if past_key_values else None,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )

    hidden_states = layer_outputs[0]
    hidden_states = self.norm(hidden_states)
    logits = self.lm_head(hidden_states)

    if not return_dict:
        return logits

    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=[layer_outputs[1]] if use_cache else None,
    )


class ModelVariant(StrEnum):
    """Available Qwen 3 EAGLE3 speculator model variants."""

    QWEN_3_8B_EAGLE3 = "8B_Eagle3"


class ModelLoader(ForgeModel):
    """Qwen 3 EAGLE3 speculator model loader for speculative decoding.

    Loads the RedHatAI Qwen3-8B EAGLE3 speculator draft model, which accelerates
    inference of the Qwen3-8B verifier model via speculative decoding.
    """

    _VARIANTS = {
        ModelVariant.QWEN_3_8B_EAGLE3: ModelConfig(
            pretrained_model_name="RedHatAI/Qwen3-8B-speculator.eagle3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_8B_EAGLE3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 Speculator EAGLE3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen 3 8B EAGLE3 speculator model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The EAGLE3 speculator model instance.
        """
        cfg = self._variant_config

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        model_kwargs |= kwargs

        model = SpeculatorModel.from_pretrained(
            cfg.pretrained_model_name,
            **model_kwargs,
        )
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.forward = types.MethodType(_eagle3_forward, model)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the EAGLE3 speculator model.

        The speculator takes hidden states from the verifier model (Qwen3-8B)
        and input token IDs.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.

        Returns:
            dict: Input tensors containing hidden states and input IDs.
        """
        dtype = dtype_override or torch.bfloat16
        hidden_size = 4096  # Qwen3-8B hidden size
        num_fused_layers = 3
        seq_len = 1

        torch.manual_seed(42)
        hidden_states = torch.randn(
            1, seq_len, num_fused_layers * hidden_size, dtype=dtype
        )
        input_ids = torch.randint(0, 151936, (1, seq_len))

        return {"input_ids": input_ids, "hidden_states": hidden_states}
