# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""VibeVoice-1.5B loader — Qwen2 language-model backbone for causal LM tasks.

microsoft/VibeVoice-1.5B is a text-to-speech model (``model_type: vibevoice``) that
is not registered in transformers and ships no pip package. Its compute-dominant
component is a standard Qwen2-1.5B decoder stored under ``model.language_model.*`` in
the checkpoint, alongside acoustic/semantic VAE tokenizers and a DDPM diffusion head
(both of which use ops unsupported by the static-shape device path). This loader
extracts only the Qwen2 backbone: it rebuilds a ``Qwen2ForCausalLM`` from the
checkpoint's ``decoder_config`` and loads the ``model.language_model.*`` weights into
it, exposing the standard causal-LM forward (logits) contract. The acoustic tokenizer,
semantic tokenizer and diffusion head are intentionally out of scope.
"""

import json
import re
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

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


class ModelVariant(StrEnum):
    """Available VibeVoice model variants."""

    VIBEVOICE_1_5B = "1.5B"


class ModelLoader(ForgeModel):
    """VibeVoice Qwen2-backbone loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.VIBEVOICE_1_5B: LLMModelConfig(
            pretrained_model_name="microsoft/VibeVoice-1.5B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_1_5B

    # The VibeVoice repo ships no tokenizer; the backbone is Qwen2 with the standard
    # Qwen2.5 vocab (151936), so we use the matching Qwen2.5-1.5B tokenizer.
    _TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"

    # Prefix of the Qwen2 backbone weights inside the VibeVoice checkpoint.
    _BACKBONE_PREFIX = "model.language_model."

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
            num_layers: Optional number of decoder layers to keep (for cheap probes).
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="VibeVoice",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load the Qwen2.5 tokenizer for the VibeVoice backbone."""
        self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _build_config(self, pretrained_model_name) -> Qwen2Config:
        """Build a Qwen2Config from the VibeVoice checkpoint's decoder_config."""
        config_path = hf_hub_download(pretrained_model_name, "config.json")
        with open(config_path) as f:
            raw = json.load(f)
        decoder_config = raw["decoder_config"]
        config = Qwen2Config(**decoder_config)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        return config

    def _load_backbone_state_dict(self, pretrained_model_name, num_layers):
        """Extract the Qwen2 backbone weights from the VibeVoice safetensors shards.

        Remaps ``model.language_model.<x>`` -> ``model.<x>`` (the key layout of a
        Qwen2ForCausalLM), downloading only the shards that hold backbone weights.
        """
        index_path = hf_hub_download(
            pretrained_model_name, "model.safetensors.index.json"
        )
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        # Group the backbone keys by the shard file that holds them.
        shard_to_keys = {}
        for key, shard in weight_map.items():
            if key.startswith(self._BACKBONE_PREFIX):
                shard_to_keys.setdefault(shard, []).append(key)

        layer_re = re.compile(r"^model\.layers\.(\d+)\.")
        state_dict = {}
        for shard, keys in shard_to_keys.items():
            shard_path = hf_hub_download(pretrained_model_name, shard)
            with safe_open(shard_path, framework="pt") as f:
                for key in keys:
                    new_key = "model." + key[len(self._BACKBONE_PREFIX) :]
                    if num_layers is not None:
                        m = layer_re.match(new_key)
                        if m and int(m.group(1)) >= num_layers:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
        return state_dict

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VibeVoice Qwen2 backbone as a Qwen2ForCausalLM."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        config = self._build_config(pretrained_model_name)
        model = Qwen2ForCausalLM(config)

        state_dict = self._load_backbone_state_dict(
            pretrained_model_name, config.num_hidden_layers
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # lm_head is tied to the input embeddings (tie_word_embeddings=True) and is
        # therefore absent from the checkpoint; tie_weights() materializes it.
        model.tie_weights()
        # The only acceptable "missing" key is the tied lm_head weight.
        unexpected_real = [k for k in missing if k != "lm_head.weight"]
        assert not unexpected_real, f"Unexpected missing backbone keys: {unexpected_real[:5]}"
        assert not unexpected, f"Unexpected extra keys: {unexpected[:5]}"

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the VibeVoice backbone."""
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
