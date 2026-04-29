# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V2 Lite GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import Qwen2MoeTensorProcessor
from typing import Optional

# deepseek_v2 uses a LLaMA-style BPE tokenizer; not registered in transformers 5.x
GGUF_TO_FAST_CONVERTERS.setdefault("deepseek_v2", GGUFLlamaConverter)

# transformers 5.x get_gguf_hf_weights_map has no deepseek_v2→deepseek2 mapping;
# gguf-py uses "deepseek2" as the arch name for DeepSeek V2 models.
# Also fix MoE expert key mapping: GGUF stores separate gate/up/down expert weights
# while the HF model expects merged gate_up_proj; the gguf-py name map returns the
# non-existent "ffn_gate_up_exps" key — replace it with the real "ffn_gate_exps" and
# "ffn_up_exps" keys so Qwen2MoeTensorProcessor can merge them.
_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map

# Reuse Qwen2MoeTensorProcessor for deepseek2: both models share the same GGUF
# expert naming scheme (ffn_gate_exps/ffn_up_exps/ffn_down_exps) and the same
# stacked gate_up_proj layout in the HF model.
_gguf_utils.TENSOR_PROCESSORS.setdefault("deepseek2", Qwen2MoeTensorProcessor)


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    if model_type is None:
        model_type = hf_model.config.model_type
    if model_type == "deepseek_v2":
        model_type = "deepseek2"
    result = _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )
    # Fix MoE expert mapping for deepseek2: gguf-py maps gate_up_proj →
    # ffn_gate_up_exps (merged), but the GGUF Q4_K_M file stores separate
    # ffn_gate_exps and ffn_up_exps.  Remove the non-existent merged key and
    # add the two real keys so Qwen2MoeTensorProcessor.process can merge them.
    if model_type == "deepseek2" and qual_name == "":
        config = hf_model.config
        num_layers_val = (
            config.num_hidden_layers if num_layers is None else num_layers
        )
        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        for i in range(first_k_dense, num_layers_val):
            merged_key = f"blk.{i}.ffn_gate_up_exps"
            hf_gate_up = result.pop(merged_key, None)
            if hf_gate_up is not None:
                result[f"blk.{i}.ffn_gate_exps"] = hf_gate_up
                result[f"blk.{i}.ffn_up_exps"] = hf_gate_up
    return result


_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

# DeepSeek V2 uses complex-valued RoPE: torch.polar creates complex freqs_cis, and
# apply_rotary_emb multiplies complex q/k with freqs_cis.  TT's ComplexDataTypeConversion
# pass decomposes stablehlo.complex (polar) and stablehlo.real/imag but does NOT
# implement the cross-product decomposition of stablehlo.multiply(complex, complex).
# Additionally, multiplying complex freqs_cis by a float scalar (attention_scaling)
# causes XLA to create a 0-dim complex constant that the TT PJRT layer rejects.
#
# Fixes applied here:
#   (a) RoPE forward: skip `* attention_scaling` when it equals 1.0 (standard case)
#       to avoid the 0-dim complex constant.
#   (b) apply_rotary_emb: replace complex-mul with real arithmetic by extracting
#       .real (cos) and .imag (sin) from freqs_cis.  stablehlo.real/imag ARE handled
#       by ComplexDataTypeConversion, so the only complex op remaining is polar→complex
#       which is also handled.
try:
    from transformers.models.deepseek_v2 import modeling_deepseek_v2 as _dsv2

    _orig_rope_forward = _dsv2.DeepseekV2YarnRotaryEmbedding.forward

    @torch.no_grad()
    def _rope_forward_no_unit_scale(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.to(x.device) @ position_ids_expanded
            ).transpose(1, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            if self.attention_scaling != 1.0:
                freqs_cis = freqs_cis * self.attention_scaling
        return freqs_cis

    _dsv2.DeepseekV2YarnRotaryEmbedding.forward = _rope_forward_no_unit_scale

    def _apply_rotary_emb_real(xq, xk, freqs_cis):
        """Real-arithmetic replacement for DeepSeek V2's complex RoPE application.

        Instead of view_as_complex → complex mul → view_as_real, extract cos/sin from
        freqs_cis using .real/.imag (lowered to stablehlo.real/imag which ComplexDataTypeConversion
        handles) and apply the rotation with real arithmetic.
        """
        cos = freqs_cis.real.unsqueeze(1)  # [batch, 1, seq, qk_rope_head_dim]
        sin = freqs_cis.imag.unsqueeze(1)

        xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
        q_even, q_odd = xq_r[..., 0], xq_r[..., 1]
        k_even, k_odd = xk_r[..., 0], xk_r[..., 1]

        xq_out = torch.stack(
            [q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1
        ).flatten(-2).type_as(xq)
        xk_out = torch.stack(
            [k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1
        ).flatten(-2).type_as(xk)
        return xq_out, xk_out

    _dsv2.apply_rotary_emb = _apply_rotary_emb_real

except (ImportError, AttributeError):
    pass

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
    """Available DeepSeek V2 Lite GGUF model variants for causal language modeling."""

    DEEPSEEK_V2_LITE_GGUF = "DeepSeek_V2_Lite_GGUF"


class ModelLoader(ForgeModel):
    """DeepSeek V2 Lite GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_V2_LITE_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/DeepSeek-V2-Lite-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_V2_LITE_GGUF

    GGUF_FILE = "DeepSeek-V2-Lite.Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="DeepSeek V2 Lite GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

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

        # Load config to potentially override architecture settings
        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        # The GGUF file stores the absorbed (merged) Q projection as "attn_q"
        # rather than the separate q_a_proj + q_a_layernorm + q_b_proj that the
        # low-rank MLA form expects.  Setting q_lora_rank=None makes the model
        # use a single q_proj that maps directly to the GGUF "attn_q" tensor.
        config.q_lora_rank = None

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
