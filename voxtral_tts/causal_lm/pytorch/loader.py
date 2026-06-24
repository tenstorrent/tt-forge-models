# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voxtral-4B-TTS loader for the language-model backbone (causal LM).

Voxtral-4B-TTS-2603 is a Mistral-native text-to-speech model (released only in
the ``vllm`` / ``mistral-common`` format, no HuggingFace ``transformers``
``AutoModel`` support). It is a multi-stage pipeline:

  * an LM backbone (a Ministral-3-3B Mistral decoder) that maps text -> audio
    semantic tokens -- the compute-dominant ~3.4B-param network,
  * a small 3-layer ``acoustic_transformer`` that predicts acoustic codebooks,
  * an ``audio_tokenizer`` codec (conv + transformer) that decodes tokens to a
    24 kHz waveform.

This loader brings up the **LM backbone** as a single-forward-pass causal LM.
The backbone is a standard Mistral decoder (GQA 32:8, head_dim 128, SwiGLU,
RMSNorm, RoPE theta=1e6, tied embeddings), so it is reconstructed as a
``MistralForCausalLM`` and the Mistral-native consolidated weights are remapped
into HuggingFace parameter names. The acoustic transformer and audio codec are
out of scope for this single-pass bringup.
"""
from typing import Optional

import torch

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
    """Available Voxtral-TTS variants."""

    VOXTRAL_4B_TTS = "4b_tts"


class ModelLoader(ForgeModel):
    """Voxtral-4B-TTS LM-backbone loader (causal language modeling)."""

    # Backbone architecture, taken from the model's params.json.
    _BACKBONE_CONFIG = dict(
        vocab_size=131072,
        hidden_size=3072,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        sliding_window=None,
        attention_dropout=0.0,
    )

    _VARIANTS = {
        ModelVariant.VOXTRAL_4B_TTS: LLMModelConfig(
            pretrained_model_name="mistralai/Voxtral-4B-TTS-2603",
            max_length=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOXTRAL_4B_TTS

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Return dashboard/reporting metadata for the given variant."""
        return ModelInfo(
            model="Voxtral-4B-TTS",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load the model's tekken tokenizer via mistral-common."""
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from huggingface_hub import hf_hub_download

        tekken_json = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="tekken.json",
        )
        self.tokenizer = MistralTokenizer.from_file(tekken_json)
        return self.tokenizer

    @staticmethod
    def _remap_backbone_state_dict(raw: dict, num_layers: int) -> dict:
        """Remap Mistral-native consolidated weights -> HF MistralForCausalLM names.

        Only the LM-backbone tensors are remapped; the ``acoustic_transformer.*``
        and ``audio_tokenizer.*`` tensors (the codec) are ignored.
        """
        tok_emb = raw["mm_audio_embeddings.tok_embeddings.weight"]
        out = {
            "model.embed_tokens.weight": tok_emb,
            "lm_head.weight": tok_emb,  # tied embeddings
            "model.norm.weight": raw["norm.weight"],
        }
        for i in range(num_layers):
            p = f"layers.{i}."
            h = f"model.layers.{i}."
            out[h + "self_attn.q_proj.weight"] = raw[p + "attention.wq.weight"]
            out[h + "self_attn.k_proj.weight"] = raw[p + "attention.wk.weight"]
            out[h + "self_attn.v_proj.weight"] = raw[p + "attention.wv.weight"]
            out[h + "self_attn.o_proj.weight"] = raw[p + "attention.wo.weight"]
            out[h + "mlp.gate_proj.weight"] = raw[p + "feed_forward.w1.weight"]
            out[h + "mlp.up_proj.weight"] = raw[p + "feed_forward.w3.weight"]
            out[h + "mlp.down_proj.weight"] = raw[p + "feed_forward.w2.weight"]
            out[h + "input_layernorm.weight"] = raw[p + "attention_norm.weight"]
            out[h + "post_attention_layernorm.weight"] = raw[p + "ffn_norm.weight"]
        return out

    def load_model(self, dtype_override=None):
        """Build the Mistral backbone and load the remapped consolidated weights.

        Args:
            dtype_override: Optional torch dtype for the model weights. Defaults
                to bfloat16 (the format the weights ship in).

        Returns:
            torch.nn.Module: MistralForCausalLM holding the Voxtral-TTS backbone.
        """
        from transformers import MistralConfig, MistralForCausalLM
        from transformers.models.mistral.modeling_mistral import (
            MistralRotaryEmbedding,
        )
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        config = MistralConfig(**self._BACKBONE_CONFIG)
        self.config = config

        # Instantiate on the meta device to skip the (slow) random init of ~3.4B
        # params; every weight is replaced by a loaded tensor via assign=True.
        with torch.device("meta"):
            model = MistralForCausalLM(config)

        ckpt = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="consolidated.safetensors",
        )
        raw = load_file(ckpt)
        state_dict = self._remap_backbone_state_dict(
            raw, config.num_hidden_layers
        )
        model.load_state_dict(state_dict, strict=False, assign=True)

        # The rotary embedding's inv_freq is a non-persistent buffer (absent from
        # the checkpoint), so it stays on the meta device after the assign-load.
        # Recreate it so it holds real data on CPU.
        model.model.rotary_emb = MistralRotaryEmbedding(config=config)

        model = model.to(dtype).eval()
        model.tie_weights()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return tokenized sample inputs (input_ids + attention_mask).

        Args:
            dtype_override: Unused for integer token inputs; kept for interface
                compatibility.
            batch_size: Batch size to replicate the sample to (default 1).

        Returns:
            dict: ``input_ids`` and ``attention_mask`` tensors.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        text = "Paris is a beautiful city, and the weather today is lovely."
        token_ids = self.tokenizer.instruct_tokenizer.tokenizer.encode(
            text, bos=True, eos=False
        )
        # Cap the sequence length to keep the device compile small.
        max_length = self._variant_config.max_length or 32
        token_ids = token_ids[:max_length]

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
