# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice-1.5B model loader implementation.

VibeVoice (microsoft/VibeVoice-1.5B) is a long-form, multi-speaker text-to-speech
model. Its compute-dominant component is a Qwen2 language-model backbone
(``decoder_config.model_type == "qwen2"``) that produces the hidden states a
diffusion head and acoustic VAE decoder turn into a 24 kHz waveform.

The published HuggingFace repo ships only weights + config (no modeling code); the
custom ``vibevoice`` package targets transformers 4.51.x and is incompatible with
the transformers version in this environment. This loader therefore brings up the
heavy-compute LLM backbone faithfully: it constructs a native ``Qwen2ForCausalLM``
from VibeVoice's ``decoder_config`` and loads the model's real, trained
``model.language_model.*`` weights from the published checkpoint shards. The
auxiliary acoustic/semantic VAE tokenizers and the 4-layer diffusion head (used
only inside the host-Python generation loop) are not part of this single-forward
bringup.
"""
import json
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available VibeVoice model variants."""

    VIBEVOICE_1_5B = "1.5B"


class ModelLoader(ForgeModel):
    """VibeVoice-1.5B loader (Qwen2 language-model backbone)."""

    _VARIANTS = {
        ModelVariant.VIBEVOICE_1_5B: LLMModelConfig(
            pretrained_model_name="microsoft/VibeVoice-1.5B",
            max_length=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_1_5B

    # VibeVoice's text tokenizer is Qwen2-based and shares Qwen2.5's 151936-token
    # vocabulary; the repo itself ships no tokenizer files, so we tokenize with the
    # matching Qwen2.5 tokenizer to feed the backbone realistic, in-distribution ids.
    TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"
    sample_text = (
        "Speaker 1: Welcome to the show, today we are talking about "
        "text to speech models and how they are brought up on hardware."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="vibevoice",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_backbone_config(self) -> Qwen2Config:
        """Build a Qwen2Config from VibeVoice's nested ``decoder_config``."""
        name = self._variant_config.pretrained_model_name
        cfg_path = hf_hub_download(name, "config.json")
        with open(cfg_path) as f:
            full_cfg = json.load(f)
        decoder_cfg = full_cfg["decoder_config"]
        return Qwen2Config(**decoder_cfg)

    def _load_language_model_weights(self, model: Qwen2ForCausalLM) -> None:
        """Load VibeVoice's real ``model.language_model.*`` weights into a Qwen2 model.

        The checkpoint stores the backbone under the ``model.language_model.`` prefix;
        remap those keys onto the native Qwen2ForCausalLM state dict (``model.``) and
        load them. Embeddings are tied, so ``lm_head`` shares ``embed_tokens``.
        """
        name = self._variant_config.pretrained_model_name
        index_path = hf_hub_download(name, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        prefix = "model.language_model."
        shards = sorted(
            {shard for key, shard in weight_map.items() if key.startswith(prefix)}
        )

        state_dict = {}
        for shard in shards:
            shard_path = hf_hub_download(name, shard)
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        # model.language_model.<x>  ->  model.<x>
                        state_dict["model." + key[len(prefix):]] = f.get_tensor(key)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # lm_head.weight is tied to embed_tokens and not stored in the checkpoint.
        assert all(
            "lm_head" in m for m in missing
        ), f"Unexpected missing weights: {missing}"
        assert not unexpected, f"Unexpected weights in checkpoint: {unexpected}"
        model.tie_weights()

    def load_model(self, dtype_override=None):
        """Load and return the VibeVoice language-model backbone.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen2ForCausalLM backbone with VibeVoice weights.
        """
        config = self._build_backbone_config()
        model = Qwen2ForCausalLM(config)
        self._load_language_model_weights(model)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the VibeVoice backbone.

        Args:
            dtype_override: Optional torch.dtype (unused; token ids stay integer).

        Returns:
            dict: ``input_ids`` and ``attention_mask`` tensors.
        """
        seq_len = self._variant_config.max_length

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
