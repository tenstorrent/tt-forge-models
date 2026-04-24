# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NBDiff-7B-Instruct model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, TypedDict

import transformers.utils as _transformers_utils
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_INIT_FUNCTIONS

# LossKwargs was removed in transformers 5.x; inject a stub so the model's
# trust_remote_code file can import it without error.
if not hasattr(_transformers_utils, "LossKwargs"):

    class _LossKwargs(TypedDict, total=False):
        labels: torch.LongTensor
        num_logits_to_keep: int

    _transformers_utils.LossKwargs = _LossKwargs


# 'default' rope type was removed from ROPE_INIT_FUNCTIONS in transformers 5.x;
# add a simple unscaled RoPE implementation as a fallback.
def _compute_default_rope_parameters(
    config=None, device=None, seq_len=None, layer_type=None
):
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
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
    return inv_freq, 1.0


if "default" not in _ROPE_INIT_FUNCTIONS:
    _ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


class _CausalMaskWrapper(torch.nn.Module):
    """Wrap a causal LM to convert 2D int attention masks to 4D float causal masks.

    PanguEmbedded (NBDiff) uses SDPA but its forward does not call create_causal_mask
    internally, so the raw tokenizer int64 mask must be pre-processed before the
    scaled_dot_product_attention call.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is not None and attention_mask.dtype not in (
            torch.bool,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            attention_mask = attention_mask.bool()
        return self._model(input_ids, attention_mask=attention_mask, **kwargs)


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
    """Available NBDiff-7B-Instruct model variants."""

    NBDIFF_7B_INSTRUCT = "nbdiff_7b_instruct"


class ModelLoader(ForgeModel):
    """NBDiff-7B-Instruct model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.NBDIFF_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="yuchuantian/NBDiff-7B-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NBDIFF_7B_INSTRUCT

    sample_text = "What is the capital of China?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NBDiff-7B-Instruct",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
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

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model_kwargs["use_cache"] = False

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = _CausalMaskWrapper(model)
        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

        if batch_size > 1:
            for i in range(len(sample_inputs)):
                sample_inputs[i] = sample_inputs[i].repeat_interleave(batch_size, dim=0)

        return sample_inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            decoded_output = self.tokenizer.decode(outputs)
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_output = self.tokenizer.decode(next_token_id)

        return decoded_output
