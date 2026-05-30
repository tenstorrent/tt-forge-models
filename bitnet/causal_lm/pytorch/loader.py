# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BitNet b1.58 (GGUF / TQ2_0) model loader for causal language modeling.

The target model ``gianni-cor/bitnet_b1_58-large-TQ2_0`` is distributed only as
a single TQ2_0-quantized GGUF file. transformers cannot load this GGUF directly
(architecture ``bitnet`` is not in its GGUF mapping, and the BitNet sub-norm
tensors have no LLaMA equivalent), so this loader:

  1. downloads the GGUF,
  2. dequantizes every tensor (TQ2_0 / Q6_K / F32) with the ``gguf`` library,
  3. loads the dequantized weights into a self-contained, inference-only BitNet
     implementation (see ``src/modeling_bitnet.py``).

The GGUF tensors are already the ternary-quantized BitLinear weights, so the
model applies only the per-token 8-bit activation fake-quant at inference and
uses the loaded weights directly.
"""

from typing import Optional

import torch
from transformers import AutoTokenizer

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
from ....tools.utils import cast_input_to_type
from .src.modeling_bitnet import BitnetConfig, BitnetForCausalLM

# GGUF tensor metadata key prefix (general.architecture == "bitnet").
_GGUF_FILENAME = "bitnet_b1_58-large-TQ2_0.gguf"
# The GGUF ships the original llama tokenizer; pull it from the base repo.
_TOKENIZER_REPO = "1bitLLM/bitnet_b1_58-large"


class ModelVariant(StrEnum):
    """Available BitNet GGUF variants for causal LM."""

    LARGE_TQ2_0 = "b1_58_large_tq2_0"


class ModelLoader(ForgeModel):
    """BitNet b1.58 (TQ2_0 GGUF) loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LARGE_TQ2_0: LLMModelConfig(
            pretrained_model_name="gianni-cor/bitnet_b1_58-large-TQ2_0",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_TQ2_0

    # Sample text for causal LM
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

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="bitnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load the tokenizer (sourced from the base BitNet repo)."""
        self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_REPO)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _build_config(self, reader) -> BitnetConfig:
        """Build a BitnetConfig from the GGUF metadata."""

        def kv(key, default):
            field = reader.fields.get(f"bitnet.{key}")
            return field.contents() if field is not None else default

        return BitnetConfig(
            vocab_size=32002,
            hidden_size=int(kv("embedding_length", 1536)),
            intermediate_size=int(kv("feed_forward_length", 4096)),
            num_hidden_layers=int(kv("block_count", 24)),
            num_attention_heads=int(kv("attention.head_count", 16)),
            num_key_value_heads=int(kv("attention.head_count_kv", 16)),
            head_dim=int(kv("attention.key_length", 96)),
            max_position_embeddings=int(kv("context_length", 2048)),
            rope_theta=float(kv("rope.freq_base", 10000.0)),
            rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon", 1e-5)),
            input_bits=8,
            tie_word_embeddings=True,
        )

    @staticmethod
    def _gguf_to_hf_name(name: str) -> Optional[str]:
        """Map a GGUF tensor name to the corresponding state-dict key."""
        if name == "token_embd.weight":
            return "model.embed_tokens.weight"
        if name == "output_norm.weight":
            return "model.norm.weight"
        if not name.startswith("blk."):
            return None
        _, idx, rest = name.split(".", 2)
        prefix = f"model.layers.{idx}"
        mapping = {
            "attn_norm.weight": f"{prefix}.input_layernorm.weight",
            "attn_q.weight": f"{prefix}.self_attn.q_proj.weight",
            "attn_k.weight": f"{prefix}.self_attn.k_proj.weight",
            "attn_v.weight": f"{prefix}.self_attn.v_proj.weight",
            "attn_output.weight": f"{prefix}.self_attn.o_proj.weight",
            "attn_sub_norm.weight": f"{prefix}.self_attn.inner_attn_ln.weight",
            "ffn_norm.weight": f"{prefix}.post_attention_layernorm.weight",
            "ffn_gate.weight": f"{prefix}.mlp.gate_proj.weight",
            "ffn_up.weight": f"{prefix}.mlp.up_proj.weight",
            "ffn_down.weight": f"{prefix}.mlp.down_proj.weight",
            "ffn_sub_norm.weight": f"{prefix}.mlp.ffn_layernorm.weight",
        }
        return mapping.get(rest)

    def load_model(self, dtype_override=None):
        """Load the BitNet model from the TQ2_0 GGUF file.

        Args:
            dtype_override: Optional torch.dtype to cast the model to.

        Returns:
            torch.nn.Module: The BitNet causal LM instance.
        """
        from gguf import GGUFReader, dequantize
        from huggingface_hub import hf_hub_download

        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_path = hf_hub_download(pretrained_model_name, _GGUF_FILENAME)
        reader = GGUFReader(gguf_path)

        config = self._build_config(reader)
        self.config = config
        model = BitnetForCausalLM(config)

        state_dict = {}
        for tensor in reader.tensors:
            hf_name = self._gguf_to_hf_name(tensor.name)
            if hf_name is None:
                continue
            arr = dequantize(tensor.data, tensor.tensor_type)
            shape = tuple(int(x) for x in reversed(tensor.shape))
            weight = torch.from_numpy(arr.astype("float32").reshape(shape))
            state_dict[hf_name] = weight

        # lm_head is tied to the embedding; provide it explicitly for safety.
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # inv_freq is a non-persistent buffer; ignore it among the missing keys.
        missing = [k for k in missing if not k.endswith("inv_freq")]
        if missing or unexpected:
            raise RuntimeError(
                f"BitNet GGUF weight load mismatch. Missing: {missing}, "
                f"Unexpected: {unexpected}"
            )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the BitNet model.

        Args:
            dtype_override: Optional torch.dtype to cast float inputs to.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: input_ids and attention_mask tensors.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")
        inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        self.seq_len = inputs["input_ids"].shape[1]

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def load_config(self):
        """Return the BitnetConfig built from the GGUF metadata."""
        if self.config is None:
            from gguf import GGUFReader
            from huggingface_hub import hf_hub_download

            gguf_path = hf_hub_download(
                self._variant_config.pretrained_model_name, _GGUF_FILENAME
            )
            self.config = self._build_config(GGUFReader(gguf_path))
        return self.config
