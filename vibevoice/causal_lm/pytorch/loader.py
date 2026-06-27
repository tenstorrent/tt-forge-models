# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice loader implementation for the Qwen2 language-model backbone.

VibeVoice (microsoft/VibeVoice-1.5B) is a multi-component text-to-speech model:
a Qwen2 decoder-only LLM backbone, conv-based acoustic and semantic VAE
tokenizers, and a small DDPM diffusion head that predicts acoustic latents.
The ``vibevoice`` model type is not registered in transformers and the checkpoint
ships no remote modeling code (the original Microsoft package was withdrawn), so
this loader brings up the compute-dominant component that runs as a single
forward pass: the Qwen2 LM backbone. Its weights live under the
``model.language_model.*`` prefix of the VibeVoice checkpoint and are standard
Qwen2 tensors (q/k/v biases included), so they are loaded into a stock
``Qwen2ForCausalLM`` built from the checkpoint's ``decoder_config``.
"""
import json
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

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
    """Available VibeVoice variants (LM backbone)."""

    V1_5B = "1.5B"


class ModelLoader(ForgeModel):
    """VibeVoice LM-backbone loader (Qwen2 decoder of the TTS pipeline)."""

    _VARIANTS = {
        ModelVariant.V1_5B: LLMModelConfig(
            pretrained_model_name="microsoft/VibeVoice-1.5B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_5B

    # Tokenizer source: VibeVoice's text LM is Qwen2.5-1.5B; the checkpoint ships
    # no tokenizer files, so use the base Qwen2.5 tokenizer for the backbone.
    _TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
            num_layers: Optional number of decoder layers (for smaller test runs).
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="VibeVoice",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load the tokenizer for the VibeVoice text LM."""
        self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)
        return self.tokenizer

    def _build_decoder_config(self) -> Qwen2Config:
        """Build a Qwen2Config from the VibeVoice checkpoint's decoder_config."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        cfg_path = hf_hub_download(pretrained_model_name, "config.json")
        with open(cfg_path) as f:
            full_cfg = json.load(f)
        decoder_cfg = dict(full_cfg["decoder_config"])
        # model_type is "qwen2"; Qwen2Config understands the remaining fields.
        decoder_cfg.pop("model_type", None)
        config = Qwen2Config(**decoder_cfg)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        return config

    def _load_backbone_state_dict(self) -> dict:
        """Download only the shards holding the LM backbone and remap their keys.

        Returns a state dict keyed for ``Qwen2ForCausalLM`` (``model.*`` /
        ``lm_head.*``) extracted from the VibeVoice ``model.language_model.*``
        tensors.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        index_path = hf_hub_download(
            pretrained_model_name, "model.safetensors.index.json"
        )
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        prefix = "model.language_model."
        shards = sorted(
            {fname for key, fname in weight_map.items() if key.startswith(prefix)}
        )

        state_dict = {}
        for shard in shards:
            shard_path = hf_hub_download(pretrained_model_name, shard)
            shard_sd = load_file(shard_path)
            for key, tensor in shard_sd.items():
                if key.startswith(prefix):
                    # model.language_model.layers.0... -> model.layers.0...
                    state_dict["model." + key[len(prefix):]] = tensor
        return state_dict

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the VibeVoice Qwen2 LM backbone.

        Args:
            dtype_override: Optional torch.dtype to cast the model to.

        Returns:
            torch.nn.Module: Qwen2ForCausalLM holding the VibeVoice LM weights.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        config = self._build_decoder_config()
        model = Qwen2ForCausalLM(config)

        state_dict = self._load_backbone_state_dict()
        # lm_head is tied to embed_tokens (tie_word_embeddings=True) and is not in
        # the checkpoint, so strict=False; tie_weights wires lm_head to the loaded
        # embeddings.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if not k.endswith("rotary_emb.inv_freq")]
        assert not unexpected, f"Unexpected VibeVoice backbone keys: {unexpected[:5]}"
        missing = [k for k in missing if k != "lm_head.weight"]
        assert not missing, f"Missing VibeVoice backbone keys: {missing[:5]}"
        model.tie_weights()

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the LM backbone.

        Tokenizes at natural length (no max-length padding) so PCC isn't
        dominated by padding positions.

        Args:
            dtype_override: Unused for integer token ids; kept for API parity.
            batch_size: Batch size for the inputs.

        Returns:
            dict: input_ids / attention_mask tensors.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            num_devices % 2 == 0
            and self.config.num_attention_heads % (num_devices // 2) == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads "
                f"across {num_devices} devices"
            )
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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Build and return the Qwen2 backbone config."""
        self.config = self._build_decoder_config()
        return self.config
