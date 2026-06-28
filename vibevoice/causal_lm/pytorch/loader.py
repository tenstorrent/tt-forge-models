# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice-1.5B loader.

microsoft/VibeVoice-1.5B is a multi-component text-to-speech model
(VibeVoiceForConditionalGeneration): a Qwen2 LM backbone + acoustic/semantic
conv-VAE tokenizers + a DDPM diffusion prediction head (~2.7B params total).

The ``vibevoice`` ``model_type`` is not present in transformers and the
checkpoint ships no remote code (the Microsoft package was withdrawn), so no
``Auto*`` class can load the full pipeline. This loader brings up the compute
component that maps to the standard transformer device path: the **Qwen2 LM
backbone**. Its weights live under the ``model.language_model.*`` prefix and are
a stock Qwen2.5-1.5B (GQA 12:2, head_dim 128, SwiGLU 8960, vocab 151936, tied
embeddings, RoPE theta 1e6). We build a Qwen2Config from the checkpoint's
``decoder_config``, instantiate ``Qwen2ForCausalLM``, load only the
``language_model.*`` tensors (remapped to the stock prefix), and tie the
lm_head. The tokenizer is borrowed from Qwen/Qwen2.5-1.5B (the checkpoint
ships none).
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available VibeVoice variants."""

    VIBEVOICE_1_5B = "1.5B"


class ModelLoader(ForgeModel):
    """VibeVoice-1.5B loader — brings up the Qwen2 LM backbone."""

    _VARIANTS = {
        ModelVariant.VIBEVOICE_1_5B: LLMModelConfig(
            pretrained_model_name="microsoft/VibeVoice-1.5B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_1_5B

    # Tokenizer source: the checkpoint ships none; the backbone is Qwen2.5-1.5B.
    _TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"

    # Prefix under which the Qwen2 backbone weights are stored in the checkpoint.
    _LM_PREFIX = "model.language_model."

    sample_text = "Hey how are you doing today?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
            num_layers: Optional number of hidden layers (for cheap sweeps).
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VibeVoice",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load the Qwen2 tokenizer used by the backbone."""
        self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _build_config(self):
        """Build a Qwen2Config from the checkpoint's decoder_config."""
        import json

        cfg_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "config.json"
        )
        with open(cfg_path) as f:
            full_cfg = json.load(f)
        decoder_cfg = full_cfg["decoder_config"]
        config = Qwen2Config(**decoder_cfg)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        return config

    def _load_backbone_state_dict(self):
        """Download the shards holding language_model.* keys and return the
        Qwen2 state dict with the prefix stripped to the stock layout."""
        repo = self._variant_config.pretrained_model_name
        index_path = hf_hub_download(repo, "model.safetensors.index.json")
        import json

        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        lm_shards = sorted(
            {v for k, v in weight_map.items() if k.startswith(self._LM_PREFIX)}
        )

        state_dict = {}
        for shard in lm_shards:
            shard_path = hf_hub_download(repo, shard)
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(self._LM_PREFIX):
                        new_key = "model." + key[len(self._LM_PREFIX) :]
                        state_dict[new_key] = f.get_tensor(key)
        return state_dict

    def load_model(self, dtype_override=None, num_layers=None, **kwargs):
        """Load and return the Qwen2 LM backbone of VibeVoice-1.5B.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.
            num_layers: Optional number of hidden layers to truncate to.

        Returns:
            torch.nn.Module: Qwen2ForCausalLM instance with VibeVoice weights.
        """
        if num_layers is not None:
            self.num_layers = num_layers
        if self.tokenizer is None:
            self._load_tokenizer()

        config = self._build_config()
        self.config = config

        model = Qwen2ForCausalLM(config)
        state_dict = self._load_backbone_state_dict()
        if self.num_layers is not None:
            # Keep only weights for the retained layers.
            keep = {}
            for k, v in state_dict.items():
                if k.startswith("model.layers."):
                    layer_idx = int(k.split(".")[2])
                    if layer_idx >= self.num_layers:
                        continue
                keep[k] = v
            state_dict = keep

        # lm_head is tied to embed_tokens; not present in the checkpoint.
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen2 backbone."""
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
