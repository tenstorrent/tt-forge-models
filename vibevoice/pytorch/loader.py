# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice model loader implementation for text-to-speech tasks.

VibeVoice (microsoft/VibeVoice-1.5B) is a long-form, multi-speaker TTS model
built as a pipeline of several networks:

  * a Qwen2 decoder LLM backbone (the language model that drives generation),
  * acoustic + semantic VAE tokenizers (conv-based speech codecs),
  * lightweight speech connectors, and
  * a small DDPM diffusion head that predicts acoustic latents conditioned on
    the LLM hidden states.

The compute-dominant and architecturally standard component is the Qwen2
decoder backbone. This loader brings up that text backbone as a single
causal-LM forward pass: input_ids -> Qwen2 backbone -> lm_head -> logits over
the 151936-token vocabulary.

The backbone weights are stored in the checkpoint under the
``model.language_model.*`` prefix in standard Qwen2 layout (with ``lm_head``
tied to the token embeddings). Rather than depend on the bespoke ``vibevoice``
package (which pins an older transformers and ships the conv VAE / diffusion
code we do not exercise here), we load just those weights into a stock
``transformers`` ``Qwen2ForCausalLM`` built from the checkpoint's
``decoder_config``. This keeps the device-bringup target a standard, fully
supported transformers architecture carrying VibeVoice's trained weights.

The speech-synthesis components (VAE codecs, diffusion head) are only exercised
during autoregressive ``generate()`` with speech inputs and are out of scope
for this single-forward-pass bringup.
"""
import json

import torch
from typing import Optional

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

    VIBEVOICE_1_5B = "1_5B"


class ModelLoader(ForgeModel):
    """VibeVoice model loader implementation for text-to-speech tasks."""

    # Tokenizer used by the Qwen2 decoder backbone (per the model's
    # preprocessor_config.json `language_model_pretrained_name`).
    _TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"

    # Prefix under which the Qwen2 backbone weights live in the checkpoint.
    _BACKBONE_PREFIX = "model.language_model."

    # Default prompt used to build sample inputs for the text backbone. Chosen
    # long enough to fill DEFAULT_SEQ_LEN with real tokens (no padding), so the
    # device-vs-CPU comparison covers only meaningful positions rather than
    # padding noise.
    DEFAULT_TEXT = (
        "Speaker 1: Hello and welcome to the show. Today we discuss the future "
        "of artificial intelligence, speech synthesis, and how large language "
        "models are changing the way people create and consume audio content."
    )

    # Default sequence length for sample inputs (kept short for a light compile).
    DEFAULT_SEQ_LEN = 32

    _VARIANTS = {
        ModelVariant.VIBEVOICE_1_5B: LLMModelConfig(
            pretrained_model_name="microsoft/VibeVoice-1.5B",
            max_length=DEFAULT_SEQ_LEN,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_1_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None

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

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            # Left-pad so the last sequence position is the final real prompt
            # token, making the next-token prediction at position -1 meaningful.
            self._tokenizer.padding_side = "left"
        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load the VibeVoice Qwen2 text backbone as a stock Qwen2ForCausalLM.

        Builds a `Qwen2ForCausalLM` from the checkpoint's `decoder_config` and
        loads the backbone weights stored under `model.language_model.*`,
        remapping them to the standard Qwen2 key layout. `lm_head` is tied to
        the token embeddings (the checkpoint ships no separate `lm_head.weight`).

        Args:
            dtype_override: Optional torch.dtype to override the model dtype.
                            Defaults to the checkpoint dtype (bfloat16).

        Returns:
            Qwen2ForCausalLM: the VibeVoice Qwen2 backbone with trained weights.
        """
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import Qwen2Config, Qwen2ForCausalLM

        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Build the Qwen2 config from the checkpoint's decoder_config.
        cfg_path = hf_hub_download(model_name, "config.json")
        with open(cfg_path) as f:
            full_cfg = json.load(f)
        decoder_cfg = dict(full_cfg["decoder_config"])
        decoder_cfg.setdefault("tie_word_embeddings", True)
        qwen_config = Qwen2Config(**decoder_cfg)

        model = Qwen2ForCausalLM(qwen_config).to(dtype)

        # Gather the backbone weights from the relevant safetensors shards,
        # remapping `model.language_model.*` -> `model.*`.
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        backbone_keys = {
            k: shard
            for k, shard in weight_map.items()
            if k.startswith(self._BACKBONE_PREFIX)
        }

        state_dict = {}
        for shard in sorted(set(backbone_keys.values())):
            shard_path = hf_hub_download(model_name, shard)
            with safe_open(shard_path, framework="pt") as f:
                for key, key_shard in backbone_keys.items():
                    if key_shard == shard:
                        new_key = "model." + key[len(self._BACKBONE_PREFIX) :]
                        state_dict[new_key] = f.get_tensor(key).to(dtype)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # `lm_head.weight` is the only expected-missing key (tied to embeddings).
        unexpected_real = [k for k in unexpected]
        missing_real = [k for k in missing if k != "lm_head.weight"]
        if unexpected_real or missing_real:
            raise RuntimeError(
                f"Unexpected weight mismatch loading VibeVoice backbone: "
                f"missing={missing_real}, unexpected={unexpected_real}"
            )
        model.tie_weights()
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, seq_len=None):
        """Load sample inputs for the VibeVoice text backbone.

        Args:
            dtype_override: Unused (inputs are integer token ids / mask).
            batch_size: Batch size for the inputs.
            seq_len: Target sequence length (padded/truncated). Defaults to
                     DEFAULT_SEQ_LEN.

        Returns:
            dict: {"input_ids", "attention_mask"} tensors of shape [batch, seq_len].
        """
        tokenizer = self._load_tokenizer()
        if seq_len is None:
            seq_len = self.DEFAULT_SEQ_LEN

        inputs = tokenizer(
            [self.DEFAULT_TEXT] * batch_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    def decode_output(self, outputs, inputs=None):
        """Decode the next-token prediction at the last position into text.

        Args:
            outputs: Model output (CausalLMOutputWithPast or a logits tensor)
                     of shape [batch, seq_len, vocab].
            inputs: Unused (kept for interface symmetry).

        Returns:
            str: The greedily predicted next-token string.
        """
        tokenizer = self._load_tokenizer()
        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        next_id = int(logits[0, -1].argmax(dim=-1))
        return tokenizer.decode([next_id])
