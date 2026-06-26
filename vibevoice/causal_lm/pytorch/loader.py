# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice-1.5B language-model backbone loader implementation.

VibeVoice (microsoft/VibeVoice-1.5B) is a multi-component text-to-speech
pipeline: a Qwen2 decoder LLM backbone, acoustic/semantic sigma-VAE tokenizers,
and a DiT-style diffusion prediction head. The whole pipeline requires the
custom (non-`transformers`) `vibevoice` package, which is not compatible with
the `transformers` 5.x in this environment.

This loader brings up the compute-dominant component — the Qwen2 decoder LLM
backbone (`model.language_model.*` weights) — as a standard causal-LM single
forward pass. The backbone weights map 1:1 onto a HuggingFace ``Qwen2ForCausalLM``
built from the checkpoint's ``decoder_config``, so no custom package is needed.
The acoustic/semantic tokenizers and diffusion head are separate components and
are not covered by this loader.
"""

import torch
from typing import Optional

from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from huggingface_hub import hf_hub_download
from safetensors import safe_open

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
    """Available VibeVoice model variants (LM backbone)."""

    VIBEVOICE_1_5B = "1.5B"


class ModelLoader(ForgeModel):
    """VibeVoice-1.5B Qwen2 LM-backbone loader for causal language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.VIBEVOICE_1_5B: LLMModelConfig(
            pretrained_model_name="microsoft/VibeVoice-1.5B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_1_5B

    # The VibeVoice acoustic/text processor reports the LM was initialized from
    # Qwen2.5-1.5B, which provides the matching tokenizer.
    TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None
        self.seq_len = None

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
            model="vibevoice",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)
        return self.tokenizer

    def load_config(self):
        """Build the Qwen2 backbone config from the VibeVoice checkpoint's
        ``decoder_config`` sub-config."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        # VibeVoice's model_type is not registered in transformers, so read the
        # raw config.json and pull out the Qwen2 `decoder_config` block.
        import json

        cfg_path = hf_hub_download(pretrained_model_name, "config.json")
        with open(cfg_path) as f:
            raw = json.load(f)
        decoder_cfg = raw["decoder_config"]
        self.config = Qwen2Config(**decoder_cfg)
        return self.config

    def load_model(self, dtype_override=None):
        """Load and return the VibeVoice Qwen2 LM-backbone instance.

        Builds a ``Qwen2ForCausalLM`` from the checkpoint's ``decoder_config``
        and loads the ``model.language_model.*`` weights (with the prefix
        remapped to ``model.*``) from the VibeVoice safetensors shards.

        Args:
            dtype_override: Optional torch.dtype to cast the model to.

        Returns:
            torch.nn.Module: The Qwen2 LM backbone for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()
        if self.config is None:
            self.load_config()

        # Build the empty Qwen2 backbone, then populate it from the VibeVoice
        # checkpoint's language_model weights.
        model = Qwen2ForCausalLM(self.config)

        state_dict = self._load_backbone_state_dict(pretrained_model_name)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # The only acceptable "missing" key is the tied lm_head (shared with
        # embed_tokens via tie_word_embeddings=True). Anything else is a bug.
        missing = [m for m in missing if m != "lm_head.weight"]
        assert not missing, f"Missing backbone weights: {missing}"
        assert not unexpected, f"Unexpected backbone weights: {unexpected}"
        model.tie_weights()

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        return model

    def _load_backbone_state_dict(self, pretrained_model_name):
        """Read only the ``model.language_model.*`` tensors from the VibeVoice
        safetensors shards and remap them to ``Qwen2ForCausalLM`` keys."""
        import json

        prefix = "model.language_model."
        index_path = hf_hub_download(
            pretrained_model_name, "model.safetensors.index.json"
        )
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        # Shards that contain at least one backbone weight.
        shards = sorted(
            {fn for key, fn in weight_map.items() if key.startswith(prefix)}
        )

        state_dict = {}
        for shard in shards:
            shard_path = hf_hub_download(pretrained_model_name, shard)
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        # model.language_model.X -> model.X (Qwen2 backbone key)
                        new_key = "model." + key[len(prefix) :]
                        state_dict[new_key] = f.get_tensor(key)
        return state_dict

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the LM backbone.

        Args:
            dtype_override: Unused for integer token inputs; kept for interface
                            compatibility.
            batch_size: Batch size for the inputs.

        Returns:
            dict: ``input_ids`` / ``attention_mask`` tensors.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        self.seq_len = max_length
        return inputs
