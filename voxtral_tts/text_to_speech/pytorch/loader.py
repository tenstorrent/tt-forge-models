# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voxtral TTS model loader implementation.

Voxtral-4B-TTS-2603 is a text-to-speech pipeline distributed in Mistral-native
format (params.json + consolidated.safetensors + tekken.json), built for
vLLM-Omni / mistral-common. It is NOT available through HuggingFace
``transformers`` Auto classes.

The pipeline has four components:
  1. LM backbone        -- a Ministral-3-3B decoder-only transformer that maps
                           text tokens to semantic/acoustic audio tokens
                           (the dominant-compute component, brought up here).
  2. acoustic_transformer -- a 3-layer depth transformer over acoustic codebooks.
  3. audio_tokenizer    -- a neural-codec decoder (conv + attention + RVQ) that
                           turns audio tokens into a 24 kHz waveform.
  4. mm_audio_embeddings -- audio-codebook + text token embeddings.

This loader brings up component (1), the LM backbone, which is architecturally a
standard Mistral ``MistralForCausalLM``. Its weights live under the ``layers.*``,
``norm`` and ``mm_audio_embeddings.tok_embeddings`` keys of
``consolidated.safetensors`` and map one-to-one onto the HF Mistral state dict.
The acoustic transformer and neural codec (the TTS-specific heads) require the
vLLM-Omni custom modeling code and contain ops outside the static-shape device
path; they are out of scope for this single-forward-pass bringup.
"""
import json
from typing import Optional

import torch
from transformers import MistralConfig, MistralForCausalLM

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Voxtral TTS model variants."""

    VOXTRAL_4B_TTS = "4b_tts"


class ModelLoader(ForgeModel):
    """Voxtral TTS LM-backbone loader (Mistral decoder-only causal LM)."""

    _VARIANTS = {
        ModelVariant.VOXTRAL_4B_TTS: LLMModelConfig(
            pretrained_model_name="mistralai/Voxtral-4B-TTS-2603",
            max_length=64,
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
        """Return model metadata for dashboard/reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        return ModelInfo(
            model="voxtral_tts",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_config(self, params: dict) -> MistralConfig:
        """Translate Voxtral params.json into a HF MistralConfig for the backbone."""
        return MistralConfig(
            vocab_size=params["vocab_size"],
            hidden_size=params["dim"],
            intermediate_size=params["hidden_dim"],
            num_hidden_layers=params["n_layers"],
            num_attention_heads=params["n_heads"],
            num_key_value_heads=params["n_kv_heads"],
            head_dim=params["head_dim"],
            hidden_act="silu",
            max_position_embeddings=params.get("max_position_embeddings", 128000),
            rms_norm_eps=params["norm_eps"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": params["rope_theta"],
            },
            sliding_window=None,
            attention_bias=params.get("use_biases", False),
            tie_word_embeddings=params.get("tied_embeddings", True),
            use_cache=False,
        )

    @staticmethod
    def _remap_backbone_state_dict(raw: dict, n_layers: int) -> dict:
        """Map Mistral-native consolidated keys onto HF MistralForCausalLM keys.

        Only the LM-backbone tensors are mapped; the acoustic transformer,
        audio tokenizer and audio-codebook embeddings are skipped.
        """
        hf = {}
        hf["model.embed_tokens.weight"] = raw["mm_audio_embeddings.tok_embeddings.weight"]
        hf["model.norm.weight"] = raw["norm.weight"]
        for i in range(n_layers):
            src = f"layers.{i}"
            dst = f"model.layers.{i}"
            hf[f"{dst}.self_attn.q_proj.weight"] = raw[f"{src}.attention.wq.weight"]
            hf[f"{dst}.self_attn.k_proj.weight"] = raw[f"{src}.attention.wk.weight"]
            hf[f"{dst}.self_attn.v_proj.weight"] = raw[f"{src}.attention.wv.weight"]
            hf[f"{dst}.self_attn.o_proj.weight"] = raw[f"{src}.attention.wo.weight"]
            # SwiGLU MLP: w1=gate, w3=up, w2=down
            hf[f"{dst}.mlp.gate_proj.weight"] = raw[f"{src}.feed_forward.w1.weight"]
            hf[f"{dst}.mlp.up_proj.weight"] = raw[f"{src}.feed_forward.w3.weight"]
            hf[f"{dst}.mlp.down_proj.weight"] = raw[f"{src}.feed_forward.w2.weight"]
            hf[f"{dst}.input_layernorm.weight"] = raw[f"{src}.attention_norm.weight"]
            hf[f"{dst}.post_attention_layernorm.weight"] = raw[
                f"{src}.ffn_norm.weight"
            ]
        return hf

    def _load_tokenizer(self):
        """Load the tekken.json tokenizer via mistral-common."""
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from huggingface_hub import hf_hub_download

        tekken_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="tekken.json",
        )
        self.tokenizer = MistralTokenizer.from_file(tekken_path)
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Voxtral TTS LM backbone.

        Builds a MistralForCausalLM matching params.json and loads the backbone
        weights from consolidated.safetensors.

        Args:
            dtype_override: Optional torch.dtype. If None, weights keep bfloat16.

        Returns:
            torch.nn.Module: The Mistral backbone for causal language modeling.
        """
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        name = self._variant_config.pretrained_model_name

        params_path = hf_hub_download(repo_id=name, filename="params.json")
        with open(params_path) as f:
            params = json.load(f)

        config = self._build_config(params)
        self.config = config

        model = MistralForCausalLM(config)

        weights_path = hf_hub_download(repo_id=name, filename="consolidated.safetensors")
        raw = {}
        with safe_open(weights_path, framework="pt") as f:
            for key in f.keys():
                if (
                    key.startswith("layers.")
                    or key == "norm.weight"
                    or key == "mm_audio_embeddings.tok_embeddings.weight"
                ):
                    raw[key] = f.get_tensor(key)

        state_dict = self._remap_backbone_state_dict(raw, config.num_hidden_layers)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Only the tied lm_head.weight should be "missing" (it is tied to
        # embed_tokens by tie_weights); nothing should be unexpected.
        assert not unexpected, f"Unexpected backbone keys: {unexpected}"
        assert all(
            "lm_head" in m for m in missing
        ), f"Unexpected missing backbone keys: {missing}"

        model.tie_weights()

        if dtype_override is not None:
            model = model.to(dtype_override)
        else:
            model = model.to(torch.bfloat16)

        model = model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return sample text token inputs for the LM backbone.

        Args:
            dtype_override: Unused for integer token inputs (kept for interface
                            compatibility).
            batch_size: Batch size to replicate the sample to.

        Returns:
            dict: input_ids / attention_mask tensors.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        test_input = "Paris is a beautiful city, and the food is incredible."

        # tekken instruct tokenizer exposes the raw byte-level tokenizer.
        token_ids = self.tokenizer.instruct_tokenizer.tokenizer.encode(
            test_input, bos=True, eos=False
        )

        max_length = self._variant_config.max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode_output(self, outputs, dtype_override=None):
        """Decode the next-token prediction from backbone logits.

        Args:
            outputs: Model output from a forward pass.

        Returns:
            str: Decoded next-token text.
        """
        if self.tokenizer is None:
            self._load_tokenizer()
        next_token = outputs.logits[:, -1].softmax(dim=-1).argmax().item()
        return self.tokenizer.instruct_tokenizer.tokenizer.decode([next_token])

    def get_mesh_config(self, num_devices: int):
        """Tensor-parallel mesh: 1 x num_devices over the model axis."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)
        assert (
            self.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        assert (
            self.config.num_key_value_heads % mesh_shape[1] == 0
        ), "KV heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron column->row tensor-parallel shard spec for the backbone."""
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
