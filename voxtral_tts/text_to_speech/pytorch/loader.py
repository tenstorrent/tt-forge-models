# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Voxtral-4B-TTS loader (text-to-speech).

Voxtral-4B-TTS is a Mistral-native checkpoint (``params.json`` +
``consolidated.safetensors`` + ``tekken.json``, distributed for vLLM, no HF
transformers ``config.json``). The full model is a multi-component TTS system:

  * a 26-layer Mistral decoder LM backbone (the ``layers.*`` + ``norm`` weights,
    based on Ministral-3-3B) that autoregressively predicts audio codebook tokens,
  * a 3-layer ``acoustic_transformer`` head,
  * a convolutional ``audio_tokenizer`` neural codec that decodes audio tokens to
    a 24 kHz waveform,
  * ``mm_audio_embeddings`` (text token embeddings + audio codebook embeddings).

This loader brings up the **LM backbone** — the compute-dominant component and the
only one that maps cleanly onto a transformers architecture. The backbone weights
are remapped from the Mistral-native key layout onto ``MistralForCausalLM`` and run
as a single forward pass over text tokens. The acoustic head and codec are
TTS-specific (conv weight-norm parametrizations, custom inference code) and are not
brought up here.
"""
import torch
from typing import Optional

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
    """Available Voxtral-TTS model variants."""

    VOXTRAL_4B_TTS = "4b_tts"


class ModelLoader(ForgeModel):
    """Loader for the Voxtral-4B-TTS Mistral LM backbone."""

    _VARIANTS = {
        ModelVariant.VOXTRAL_4B_TTS: LLMModelConfig(
            pretrained_model_name="mistralai/Voxtral-4B-TTS-2603",
            max_length=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOXTRAL_4B_TTS

    # Architecture from the checkpoint's params.json (Ministral-3-3B based).
    _DIM = 3072
    _N_LAYERS = 26
    _N_HEADS = 32
    _N_KV_HEADS = 8
    _HEAD_DIM = 128
    _HIDDEN_DIM = 9216
    _VOCAB_SIZE = 131072
    _ROPE_THETA = 1000000.0
    _NORM_EPS = 1e-5
    _MAX_POS = 128000

    _SAMPLE_TEXT = "Hello, this is a Voxtral text to speech bringup test."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="voxtral_tts",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_config(self):
        from transformers import MistralConfig

        return MistralConfig(
            hidden_size=self._DIM,
            num_hidden_layers=self._N_LAYERS,
            num_attention_heads=self._N_HEADS,
            num_key_value_heads=self._N_KV_HEADS,
            head_dim=self._HEAD_DIM,
            intermediate_size=self._HIDDEN_DIM,
            vocab_size=self._VOCAB_SIZE,
            max_position_embeddings=self._MAX_POS,
            rms_norm_eps=self._NORM_EPS,
            # transformers 5.x uses rope_parameters, not rope_theta.
            rope_parameters={"rope_type": "default", "rope_theta": self._ROPE_THETA},
            hidden_act="silu",
            attention_bias=False,
            tie_word_embeddings=True,
            sliding_window=None,
            use_cache=False,
        )

    @staticmethod
    def _remap_key(k: str):
        """Map a Mistral-native checkpoint key to a MistralForCausalLM key.

        Returns the target key, or None if the tensor belongs to an audio
        component not part of the LM backbone.
        """
        if k == "norm.weight":
            return "model.norm.weight"
        if k == "mm_audio_embeddings.tok_embeddings.weight":
            return "model.embed_tokens.weight"
        if k.startswith("layers."):
            parts = k.split(".")
            idx = parts[1]
            base = f"model.layers.{idx}"
            sub = ".".join(parts[2:])
            mapping = {
                "attention.wq.weight": "self_attn.q_proj.weight",
                "attention.wk.weight": "self_attn.k_proj.weight",
                "attention.wv.weight": "self_attn.v_proj.weight",
                "attention.wo.weight": "self_attn.o_proj.weight",
                "attention_norm.weight": "input_layernorm.weight",
                "ffn_norm.weight": "post_attention_layernorm.weight",
                "feed_forward.w1.weight": "mlp.gate_proj.weight",
                "feed_forward.w3.weight": "mlp.up_proj.weight",
                "feed_forward.w2.weight": "mlp.down_proj.weight",
            }
            if sub in mapping:
                return f"{base}.{mapping[sub]}"
        # acoustic_transformer.*, audio_tokenizer.*, audio_codebook_embeddings -> skip
        return None

    def load_model(self, dtype_override=None):
        from transformers import MistralForCausalLM
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        config = self._build_config()
        model = MistralForCausalLM(config)

        # Pull only the backbone tensors out of the Mistral-native checkpoint.
        weights_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "consolidated.safetensors"
        )
        state_dict = {}
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for src in f.keys():
                dst = self._remap_key(src)
                if dst is not None:
                    state_dict[dst] = f.get_tensor(src).to(dtype)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # lm_head.weight is tied to embed_tokens; it is the only acceptable "missing".
        real_missing = [m for m in missing if m != "lm_head.weight"]
        assert not real_missing, f"Unmapped backbone params: {real_missing[:8]}"
        assert not unexpected, f"Unexpected mapped params: {unexpected[:8]}"

        model = model.to(dtype)
        model.eval()
        return model

    def _load_tokenizer(self):
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        from huggingface_hub import hf_hub_download

        tekken_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "tekken.json"
        )
        self.tokenizer = Tekkenizer.from_file(tekken_path)
        return self.tokenizer

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        seq_len = self._variant_config.max_length
        ids = self.tokenizer.encode(self._SAMPLE_TEXT, bos=True, eos=False)
        # Static shape: pad/trim to seq_len.
        if len(ids) < seq_len:
            ids = ids + [0] * (seq_len - len(ids))
        else:
            ids = ids[:seq_len]

        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_mesh_config(self, num_devices: int):
        """Tensor-parallel mesh: a 1×N model axis (Megatron column→row)."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)
        assert (
            self._N_HEADS % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        assert (
            self._N_KV_HEADS % mesh_shape[1] == 0
        ), "KV heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron tensor-parallel plan: attention/MLP column-sharded on the
        "model" axis, output projections row-sharded."""
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
