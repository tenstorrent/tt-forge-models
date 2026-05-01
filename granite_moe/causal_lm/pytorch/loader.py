# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite MoE model loader implementation for causal language modeling.
"""
import types
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patched_get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
    """Fix f64 promotion: replace Python float literals with dtype-typed tensors.

    The upstream implementation uses `1.0` and `torch.finfo(dtype).min` as Python
    floats, which XLA traces as float64 constants and promotes the attention mask
    to f64. TT hardware cannot handle f64. Patch keeps computation in model dtype.
    """
    if dtype is None:
        dtype = self.dtype
    if attention_mask.dim() == 3:
        extended = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        from transformers.modeling_utils import ModuleUtilsMixin
        if getattr(self.config, "is_decoder", None):
            extended = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask
            )
        else:
            extended = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    extended = extended.to(dtype=dtype)
    one = torch.tensor(1.0, dtype=dtype)
    min_val = torch.tensor(torch.finfo(dtype).min, dtype=dtype)
    return (one - extended) * min_val


def _patched_topk_gating_forward(self, hidden_states):
    """Avoids expert_size.tolist() by returning sorted_expert_ids as a tensor.

    GraniteMoeTopKGating.forward calls expert_size.tolist() which triggers a
    device-to-host transfer that fails on TT silicon (INTERNAL error code 13).
    Return sorted_expert_ids as an int32 tensor so the caller can use it with
    masked matmul instead of dynamic split.
    """
    logits = self.layer(hidden_states).float()
    top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
    top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)

    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")

    sorted_expert_ids = top_k_experts[index_sorted_experts].int()

    top_k_gates = top_k_gates.flatten()
    batch_gates = top_k_gates[index_sorted_experts]

    return index_sorted_experts, batch_index, batch_gates, sorted_expert_ids, logits


def _patched_parallel_experts_forward(self, inputs, sorted_expert_ids):
    """Uses per-expert masked matmul instead of split-by-expert-size.

    GraniteMoeParallelExperts.forward calls inputs.split(expert_size) with a
    Python list, requiring a device-to-host transfer. For each expert e,
    compute F.linear for all tokens and zero-out tokens not assigned to e via
    a boolean mask. All ops stay in tensor-land with no device-to-host transfers.
    """
    T = inputs.shape[0]
    result = torch.zeros(T, self.output_size, dtype=inputs.dtype, device=inputs.device)
    for e in range(self.num_experts):
        w_e = self.weight[e]
        out_e = F.linear(inputs, w_e)
        mask_e = (sorted_expert_ids == e).to(inputs.dtype).unsqueeze(1)
        result = result + out_e * mask_e
    return result


def _patched_eager_mask(
    batch_size,
    cache_position,
    kv_length,
    kv_offset=0,
    mask_function=None,
    attention_mask=None,
    dtype=torch.float32,
    allow_is_bidirectional_skip=False,
    use_vmap=False,
    **kwargs,
):
    """Fix f64 promotion: wrap min_dtype as a dtype-typed tensor in torch.where.

    The upstream eager_mask passes torch.finfo(dtype).min as a Python float scalar
    to torch.where, which XLA traces as a float64 constant and promotes the mask
    to f64. TT hardware cannot handle f64. This patch keeps the mask in model dtype.
    """
    from transformers.masking_utils import sdpa_mask, causal_mask_function

    if mask_function is None:
        mask_function = causal_mask_function
    kwargs.pop("allow_is_causal_skip", None)
    kwargs.pop("allow_torch_fix", None)
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        allow_torch_fix=False,
        use_vmap=use_vmap,
        **kwargs,
    )
    if mask is not None:
        min_val = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=mask.device)
        mask = torch.where(
            mask,
            torch.tensor(0.0, dtype=dtype, device=mask.device),
            min_val,
        )
    return mask


def _patch_moe_experts(model):
    from transformers.models.granitemoe.modeling_granitemoe import (
        GraniteMoeParallelExperts,
        GraniteMoeTopKGating,
    )

    for module in model.modules():
        if isinstance(module, GraniteMoeTopKGating):
            module.forward = types.MethodType(_patched_topk_gating_forward, module)
        elif isinstance(module, GraniteMoeParallelExperts):
            module.forward = types.MethodType(_patched_parallel_experts_forward, module)

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
    """Available Granite MoE model variants for causal language modeling."""

    GRANITE_3_0_1B_A400M_BASE = "3.0_1B_A400M_Base"
    GRANITE_3_1_1B_A400M_BASE = "3.1_1B_A400M_Base"


class ModelLoader(ForgeModel):
    """Granite MoE model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_3_0_1B_A400M_BASE: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-3.0-1b-a400m-base",
            max_length=128,
        ),
        ModelVariant.GRANITE_3_1_1B_A400M_BASE: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-3.1-1b-a400m-base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_3_0_1B_A400M_BASE

    sample_text = "Where is the Thomas J. Watson Research Center located?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Granite MoE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        _patch_moe_experts(model)
        model.get_extended_attention_mask = types.MethodType(
            _patched_get_extended_attention_mask, model
        )

        import transformers.masking_utils as _mu
        _mu.AttentionMaskInterface._global_mapping["eager"] = _patched_eager_mask

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
