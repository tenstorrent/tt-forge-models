# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sabafallah DeepSeek OCR GGUF model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_transformers_deepseek2_ocr_gguf():
    """Monkey-patch transformers to add deepseek2-ocr GGUF architecture support.

    deepseek2-ocr uses standard MHA (not MLA) with DeepSeek-V2-style MoE FFN.
    Key differences from deepseek2:
    - rope.dimension_count == 0 (no RoPE), stored as qk_rope_head_dim in HF config
    - attention uses attn_q/k/v tensors (standard projections), not MLA tensors
    - config.head_dim = qk_rope_head_dim = 0, which causes `0 or hidden//heads`
      to incorrectly produce a non-zero rotary embedding dimension

    Fixes applied:
    1. Register deepseek2-ocr as a supported GGUF architecture, aliased to deepseek_v2
    2. Set q_lora_rank=None so the standard q_proj path is used (matches attn_q tensor)
    3. Patch DeepseekV2RotaryEmbedding to return zero-length inv_freq when
       qk_rope_head_dim==0, preventing a shape mismatch in apply_rotary_emb
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    _ARCH = "deepseek2-ocr"

    if _ARCH not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append(_ARCH)

        GGUF_TO_TRANSFORMERS_MAPPING["config"][_ARCH] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.freq_base": "rope_theta",
            "rope.dimension_count": "qk_rope_head_dim",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": None,
            "attention.value_length": None,
            "attention.key_length_mla": "qk_nope_head_dim",
            "attention.value_length_mla": "v_head_dim",
            "attention.q_lora_rank": "q_lora_rank",
            "attention.kv_lora_rank": "kv_lora_rank",
            "vocab_size": "vocab_size",
            "expert_count": "n_routed_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_shared_count": "n_shared_experts",
            "expert_group_count": "n_group",
            "expert_group_used_count": "topk_group",
            "expert_weights_scale": "routed_scaling_factor",
            "expert_weights_norm": "norm_topk_prob",
            "leading_dense_block_count": "first_k_dense_replace",
            "expert_feed_forward_length": "moe_intermediate_size",
        }

        orig_load = gguf_utils.load_gguf_checkpoint

        def _patched_load_gguf_checkpoint(*args, **kwargs):
            result = orig_load(*args, **kwargs)
            config = result.get("config", {})
            if config.get("model_type") == _ARCH:
                config["model_type"] = "deepseek_v2"
                # This model uses standard Q projection (no LoRA compression);
                # without this override the default q_lora_rank=1536 would be
                # used, creating q_a_proj/q_b_proj layers that don't match the
                # attn_q GGUF tensor.
                config.setdefault("q_lora_rank", None)
            return result

        gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

        import transformers.models.auto.tokenization_auto as tok_auto
        import transformers.configuration_utils as config_utils
        import transformers.modeling_utils as modeling_utils

        for mod in (tok_auto, config_utils, modeling_utils):
            if hasattr(mod, "load_gguf_checkpoint"):
                mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if _ARCH not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS[_ARCH] = GGUFQwen2Converter
    if "deepseek_v2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter

    import transformers.modeling_gguf_pytorch_utils as _gguf_utils2

    orig_get_map = _gguf_utils2.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("deepseek_v2", _ARCH):
            model_type = "deepseek2"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils2.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

    # Fix: when qk_rope_head_dim==0, DeepseekV2RotaryEmbedding.compute_default_rope_parameters
    # uses `0 or hidden_size//num_heads` (Python treats 0 as falsy), producing a
    # rotary embedding of the full head dim.  apply_rotary_emb then fails because
    # q_pe has shape [..., 0] but freqs_cis has shape [..., head_dim/2].
    # Patch to return a zero-length inv_freq when the rope dimension is 0.
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2RotaryEmbedding,
    )

    _orig_compute_rope = DeepseekV2RotaryEmbedding.compute_default_rope_parameters

    @staticmethod
    def _patched_compute_rope(config, device=None, seq_len=None):
        rope_dim = getattr(config, "qk_rope_head_dim", None)
        if rope_dim == 0:
            return torch.zeros(0, dtype=torch.float32, device=device), 1.0
        return _orig_compute_rope(config, device, seq_len)

    DeepseekV2RotaryEmbedding.compute_default_rope_parameters = _patched_compute_rope


_patch_transformers_deepseek2_ocr_gguf()


class ModelVariant(StrEnum):
    """Available Sabafallah DeepSeek OCR GGUF model variants for causal language modeling."""

    SABAFALLAH_DEEPSEEK_OCR_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Sabafallah DeepSeek OCR GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SABAFALLAH_DEEPSEEK_OCR_Q4_K_M: LLMModelConfig(
            pretrained_model_name="sabafallah/DeepSeek-OCR-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SABAFALLAH_DEEPSEEK_OCR_Q4_K_M

    GGUF_FILE = "deepseek-ocr-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="Sabafallah DeepSeek OCR GGUF",
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
