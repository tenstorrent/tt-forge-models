# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voxtral TTS loader implementation for the language-model backbone.

mistralai/Voxtral-4B-TTS-2603 is a text-to-speech model distributed in Mistral's
native format (``params.json`` + ``consolidated.safetensors`` + ``tekken.json``),
not the HuggingFace ``transformers`` format. The full model is a multi-stage
pipeline:

  * an LM backbone  -- a Ministral-3-3B-derived decoder transformer
    (26 layers, GQA 32:8, head_dim 128, SwiGLU FFN, RMSNorm, tied embeddings),
    weights under ``layers.*`` / ``mm_audio_embeddings.tok_embeddings`` / ``norm``;
  * an ``acoustic_transformer`` -- a small 3-layer transformer that emits the
    semantic / acoustic codebook tokens;
  * an ``audio_tokenizer``       -- a convolutional neural audio codec that turns
    those tokens into a 24 kHz waveform.

Only the LM backbone has a faithful, public ``torch`` implementation (it is a
plain Mistral decoder); the acoustic transformer and neural codec have no public
``torch`` modeling code -- the reference runtime is ``vllm-omni``. Following the
composite-component bringup methodology, this loader brings up the
compute-dominant LM backbone: it reconstructs a ``MistralForCausalLM`` from
``params.json`` and loads the backbone tensors out of ``consolidated.safetensors``
with the standard Mistral-native -> HF weight remap (including the RoPE permute
on the q/k projections).
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
    """Available Voxtral TTS variants."""

    # The LM backbone of the 4B TTS model (Ministral-3-3B-derived decoder).
    BACKBONE_4B = "4b_lm_backbone"


class ModelLoader(ForgeModel):
    """Voxtral-4B-TTS language-model backbone loader."""

    _VARIANTS = {
        ModelVariant.BACKBONE_4B: LLMModelConfig(
            pretrained_model_name="mistralai/Voxtral-4B-TTS-2603",
            max_length=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BACKBONE_4B

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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="VoxtralTTS",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # ------------------------------------------------------------------
    # Weight conversion helpers (Mistral-native -> HF transformers)
    # ------------------------------------------------------------------
    @staticmethod
    def _permute_rope(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        """Permute a q/k projection from Mistral-native interleaved RoPE layout
        to the HF ``rotate_half`` layout (the standard Mistral checkpoint
        conversion permute)."""
        out_dim, in_dim = w.shape  # [n_heads * head_dim, hidden]
        return (
            w.view(n_heads, head_dim // 2, 2, in_dim)
            .transpose(1, 2)
            .reshape(out_dim, in_dim)
        )

    def _build_config(self, params: dict):
        """Build a MistralConfig for the LM backbone from Mistral params.json."""
        from transformers import MistralConfig

        cfg = MistralConfig(
            vocab_size=params["vocab_size"],
            hidden_size=params["dim"],
            intermediate_size=params["hidden_dim"],
            num_hidden_layers=params["n_layers"],
            num_attention_heads=params["n_heads"],
            num_key_value_heads=params["n_kv_heads"],
            head_dim=params["head_dim"],
            max_position_embeddings=params.get("max_position_embeddings", 128000),
            rms_norm_eps=params["norm_eps"],
            rope_theta=params["rope_theta"],
            hidden_act="silu",
            tie_word_embeddings=params.get("tied_embeddings", True),
            sliding_window=None,  # backbone uses full causal attention
            attention_bias=params.get("use_biases", False),
            mlp_bias=params.get("use_biases", False),
            use_cache=False,
        )
        return cfg

    def _remap_state_dict(self, f, cfg) -> dict:
        """Remap backbone tensors from the open safetensors file ``f`` into an
        HF MistralForCausalLM state dict."""
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.head_dim

        sd = {}
        # Token embeddings (tied to the LM head).
        embed = f.get_tensor("mm_audio_embeddings.tok_embeddings.weight")
        sd["model.embed_tokens.weight"] = embed
        # Preserve weight tying by sharing the same tensor object.
        sd["lm_head.weight"] = embed

        for i in range(cfg.num_hidden_layers):
            src = f"layers.{i}"
            dst = f"model.layers.{i}"
            wq = f.get_tensor(f"{src}.attention.wq.weight")
            wk = f.get_tensor(f"{src}.attention.wk.weight")
            sd[f"{dst}.self_attn.q_proj.weight"] = self._permute_rope(
                wq, n_heads, head_dim
            )
            sd[f"{dst}.self_attn.k_proj.weight"] = self._permute_rope(
                wk, n_kv_heads, head_dim
            )
            sd[f"{dst}.self_attn.v_proj.weight"] = f.get_tensor(
                f"{src}.attention.wv.weight"
            )
            sd[f"{dst}.self_attn.o_proj.weight"] = f.get_tensor(
                f"{src}.attention.wo.weight"
            )
            sd[f"{dst}.mlp.gate_proj.weight"] = f.get_tensor(
                f"{src}.feed_forward.w1.weight"
            )
            sd[f"{dst}.mlp.down_proj.weight"] = f.get_tensor(
                f"{src}.feed_forward.w2.weight"
            )
            sd[f"{dst}.mlp.up_proj.weight"] = f.get_tensor(
                f"{src}.feed_forward.w3.weight"
            )
            sd[f"{dst}.input_layernorm.weight"] = f.get_tensor(
                f"{src}.attention_norm.weight"
            )
            sd[f"{dst}.post_attention_layernorm.weight"] = f.get_tensor(
                f"{src}.ffn_norm.weight"
            )

        sd["model.norm.weight"] = f.get_tensor("norm.weight")
        return sd

    def _load_tokenizer(self):
        """Load the tekken tokenizer shipped with the model (via mistral-common)."""
        from huggingface_hub import hf_hub_download
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tekken_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="tekken.json",
        )
        self.tokenizer = MistralTokenizer.from_file(tekken_path)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Voxtral TTS LM backbone as a MistralForCausalLM.

        The backbone weights are read out of the Mistral-native
        ``consolidated.safetensors`` and remapped into the HF layout.

        Args:
            dtype_override: Optional torch.dtype. Defaults to bfloat16.

        Returns:
            torch.nn.Module: MistralForCausalLM backbone instance.
        """
        import json

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import MistralForCausalLM
        from transformers.models.mistral.modeling_mistral import (
            MistralRotaryEmbedding,
        )

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        repo = self._variant_config.pretrained_model_name

        params_path = hf_hub_download(repo_id=repo, filename="params.json")
        with open(params_path) as fh:
            params = json.load(fh)

        cfg = self._build_config(params)
        self.config = cfg

        weights_path = hf_hub_download(repo_id=repo, filename="consolidated.safetensors")

        # Construct on the meta device (instant, no random-init of 3.8B params),
        # then materialize every parameter directly from the loaded tensors.
        with torch.device("meta"):
            model = MistralForCausalLM(cfg)

        with safe_open(weights_path, framework="pt", device="cpu") as f:
            state_dict = self._remap_state_dict(f, cfg)
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
            # assign=True adopts the loaded tensors as the model parameters,
            # replacing the meta placeholders.
            model.load_state_dict(state_dict, strict=True, assign=True)

        # The RoPE inv_freq buffers are non-persistent (not in the state dict),
        # so they are still on meta after the assign-load -- rebuild them on CPU.
        model.model.rotary_emb = MistralRotaryEmbedding(config=cfg, device="cpu")

        model = model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs (input_ids, attention_mask).

        Args:
            dtype_override: Unused for integer token inputs; accepted for API
                parity with other loaders.
            batch_size: Batch size to replicate the prompt across.

        Returns:
            dict: {"input_ids", "attention_mask"} tensors for the backbone.
        """
        seq_len = self._variant_config.max_length or 32
        prompt = (
            "Paris is a beautiful city, and the Voxtral text to speech model "
            "speaks many languages."
        )

        token_ids = None
        try:
            if self.tokenizer is None:
                self._load_tokenizer()
            # Low-level tekken text encode (avoids chat-template machinery).
            token_ids = self.tokenizer.instruct_tokenizer.tokenizer.encode(
                prompt, bos=True, eos=False
            )
        except Exception:
            token_ids = None

        if not token_ids:
            # Deterministic fallback if the tekken tokenizer API is unavailable;
            # any in-vocab ids exercise the backbone forward equivalently.
            token_ids = [1] + list(range(100, 100 + seq_len - 1))

        # Pad / truncate to a fixed sequence length for a stable compile shape.
        if len(token_ids) >= seq_len:
            token_ids = token_ids[:seq_len]
        else:
            token_ids = token_ids + [0] * (seq_len - len(token_ids))

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def load_config(self):
        """Return the backbone MistralConfig (loading the model if needed)."""
        if self.config is None:
            self.load_model()
        return self.config
