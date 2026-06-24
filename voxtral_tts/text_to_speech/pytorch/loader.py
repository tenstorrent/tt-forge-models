# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voxtral-4B-TTS model loader implementation.

Voxtral-4B-TTS-2603 is a Mistral-native (consolidated.safetensors / params.json)
text-to-speech model. It is a multi-component pipeline:

  * a ~3B-parameter Mistral decoder text backbone (``layers.*`` + token
    embeddings + final norm) — the compute-dominant component,
  * a 3-layer acoustic transformer, and
  * a neural audio-codec tokenizer (conv + attention decoder blocks).

The model ships in Mistral's native format (no HuggingFace ``config.json``) and
its runtime lives in vLLM / mistral-common. For bringup we load the
compute-dominant text backbone as a standard ``MistralForCausalLM``: we build a
matching ``MistralConfig`` from ``params.json`` and remap the mistral-native
weight names (and apply the canonical mistral->HF RoPE permutation to the q/k
projections) onto the HuggingFace Mistral state dict. This validates a single
forward pass of the dominant network on CPU and on Tenstorrent hardware.
"""
import torch
from typing import Optional

from transformers import MistralConfig, MistralForCausalLM, AutoTokenizer

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
    """Available Voxtral-TTS model variants."""

    VOXTRAL_4B_TTS = "4b_tts"


class ModelLoader(ForgeModel):
    """Voxtral-4B-TTS loader — brings up the Mistral text backbone."""

    # Text-backbone hyper-parameters, taken verbatim from the model's params.json.
    _BACKBONE_HPARAMS = dict(
        hidden_size=3072,
        num_hidden_layers=26,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=9216,
        vocab_size=131072,
        max_position_embeddings=128000,
        rope_theta=1000000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        hidden_act="silu",
        sliding_window=None,
    )

    # The TTS checkpoint ships tekken.json (no HF tokenizer). Its base model does
    # ship a HF tokenizer with the identical 131072-token vocabulary, which we
    # reuse to build sample input_ids.
    _TOKENIZER_NAME = "mistralai/Ministral-3-3B-Base-2512"

    _VARIANTS = {
        ModelVariant.VOXTRAL_4B_TTS: LLMModelConfig(
            pretrained_model_name="mistralai/Voxtral-4B-TTS-2603",
            max_length=128,
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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

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

    @staticmethod
    def _permute_for_rope(weight: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Apply the canonical mistral-native -> HuggingFace RoPE permutation.

        HF's Mistral uses ``rotate_half`` (split into two contiguous halves),
        while the mistral-native checkpoint interleaves the rotary pairs. This
        is the same permutation used by transformers' convert_mistral_weights.
        """
        d1, d2 = weight.shape
        return (
            weight.view(n_heads, d1 // n_heads // 2, 2, d2)
            .transpose(1, 2)
            .reshape(d1, d2)
        )

    def _remap_state_dict(self, safetensors_path: str) -> dict:
        """Build a HuggingFace MistralForCausalLM state dict from the
        mistral-native consolidated.safetensors of the TTS checkpoint."""
        from safetensors import safe_open

        n_heads = self._BACKBONE_HPARAMS["num_attention_heads"]
        n_kv = self._BACKBONE_HPARAMS["num_key_value_heads"]
        n_layers = self._BACKBONE_HPARAMS["num_hidden_layers"]

        sd = {}
        with safe_open(safetensors_path, framework="pt") as f:
            # Token embeddings live under mm_audio_embeddings; final norm is top-level.
            sd["model.embed_tokens.weight"] = f.get_tensor(
                "mm_audio_embeddings.tok_embeddings.weight"
            )
            sd["model.norm.weight"] = f.get_tensor("norm.weight")
            for i in range(n_layers):
                src = f"layers.{i}."
                dst = f"model.layers.{i}."
                sd[dst + "self_attn.q_proj.weight"] = self._permute_for_rope(
                    f.get_tensor(src + "attention.wq.weight"), n_heads
                )
                sd[dst + "self_attn.k_proj.weight"] = self._permute_for_rope(
                    f.get_tensor(src + "attention.wk.weight"), n_kv
                )
                sd[dst + "self_attn.v_proj.weight"] = f.get_tensor(
                    src + "attention.wv.weight"
                )
                sd[dst + "self_attn.o_proj.weight"] = f.get_tensor(
                    src + "attention.wo.weight"
                )
                sd[dst + "mlp.gate_proj.weight"] = f.get_tensor(
                    src + "feed_forward.w1.weight"
                )
                sd[dst + "mlp.down_proj.weight"] = f.get_tensor(
                    src + "feed_forward.w2.weight"
                )
                sd[dst + "mlp.up_proj.weight"] = f.get_tensor(
                    src + "feed_forward.w3.weight"
                )
                sd[dst + "input_layernorm.weight"] = f.get_tensor(
                    src + "attention_norm.weight"
                )
                sd[dst + "post_attention_layernorm.weight"] = f.get_tensor(
                    src + "ffn_norm.weight"
                )
        return sd

    def _load_tokenizer(self):
        """Load the (base-model) HF tokenizer matching the TTS vocab."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._TOKENIZER_NAME, padding_side="right"
        )
        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Voxtral-TTS text backbone as MistralForCausalLM.

        Args:
            dtype_override: Optional torch.dtype to override the default bfloat16.

        Returns:
            torch.nn.Module: The Mistral text-backbone model instance.
        """
        from huggingface_hub import hf_hub_download

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        config = MistralConfig(torch_dtype=dtype, **self._BACKBONE_HPARAMS)
        self.config = config

        safetensors_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "consolidated.safetensors"
        )
        state_dict = self._remap_state_dict(safetensors_path)

        model = MistralForCausalLM(config).to(dtype)
        # lm_head.weight is tied to embed_tokens and so absent from state_dict;
        # everything else must map exactly.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert not unexpected, f"Unexpected weights: {unexpected[:5]}"
        assert all(
            m == "lm_head.weight" for m in missing
        ), f"Unexpected missing weights: {missing[:5]}"

        model = model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Voxtral-TTS text backbone.

        Args:
            dtype_override: Unused for token inputs (kept for interface parity).
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask).
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        test_input = "Generate speech for the following text: Hello and welcome."
        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    def decode_output(self, outputs, dtype_override=None):
        """Decode the next-token logits into a token id / string.

        Note: this is the audio-conditioned TTS backbone, so the next token is
        typically an audio-codebook token rather than coherent text.
        """
        if self.tokenizer is None:
            self._load_tokenizer()
        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.float().softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
