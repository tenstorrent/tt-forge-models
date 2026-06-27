# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice causal LM loader implementation.

VibeVoice (microsoft/VibeVoice-1.5B) is a long-form text-to-speech pipeline made
of several components: a Qwen2 language-model backbone, a DDPM diffusion
"prediction head", and acoustic/semantic VAE tokenizers. The compute-dominant
component (~1.5B params) is the Qwen2 backbone stored under the
``model.language_model.*`` keys of the checkpoint.

The full pipeline relies on Microsoft's custom modeling code (model_type
``vibevoice`` is not part of upstream ``transformers``). The backbone, however, is
a standard Qwen2 decoder whose weights map 1:1 onto ``Qwen2ForCausalLM``. This
loader therefore reconstructs the Qwen2 backbone directly from the checkpoint's
``decoder_config`` and the ``model.language_model.*`` weight subset, so the
single-forward-pass bringup needs no custom package. The diffusion head and VAE
tokenizers are out of scope for this loader (see the bringup report).
"""

import torch
from typing import Optional

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
    """Available VibeVoice model variants for the Qwen2 backbone."""

    VIBEVOICE_1_5B = "1.5B"


class ModelLoader(ForgeModel):
    """VibeVoice loader: the Qwen2 language-model backbone for causal LM tasks."""

    # The checkpoint ships its own tokenizer files implicitly via the Qwen2.5
    # vocabulary (vocab_size 151936). VibeVoice-1.5B does not bundle tokenizer
    # files, so we use the matching Qwen2.5-1.5B tokenizer for input prep.
    _TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.VIBEVOICE_1_5B: LLMModelConfig(
            pretrained_model_name="microsoft/VibeVoice-1.5B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_1_5B

    # Shared configuration parameters
    sample_text = "Hello, welcome to today's podcast about large language models."

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

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="VibeVoice",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer (Qwen2.5-1.5B vocab) for the current variant."""
        self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)
        return self.tokenizer

    def _load_backbone_state_dict(self, pretrained_model_name):
        """Download the checkpoint shards and extract the Qwen2 backbone weights.

        Returns the ``model.language_model.*`` weights with that prefix stripped
        so they can be loaded into a ``Qwen2ForCausalLM`` (under its ``model.``
        prefix).
        """
        import json
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        prefix = "model.language_model."

        index_path = hf_hub_download(
            pretrained_model_name, "model.safetensors.index.json"
        )
        weight_map = json.load(open(index_path))["weight_map"]
        shards = sorted(
            {shard for key, shard in weight_map.items() if key.startswith(prefix)}
        )

        state_dict = {}
        for shard in shards:
            shard_path = hf_hub_download(pretrained_model_name, shard)
            tensors = load_file(shard_path)
            for key, tensor in tensors.items():
                if key.startswith(prefix):
                    state_dict[key[len(prefix) :]] = tensor
        return state_dict

    def load_model(self, dtype_override=None):
        """Load and return the VibeVoice Qwen2 backbone for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. The checkpoint is distributed in bfloat16.

        Returns:
            torch.nn.Module: The Qwen2ForCausalLM backbone instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        # Build the Qwen2 config from the checkpoint's decoder_config block.
        full_config = Qwen2Config.get_config_dict(pretrained_model_name)[0]
        decoder_config = full_config["decoder_config"]
        config = Qwen2Config(**decoder_config)
        self.config = config

        model = Qwen2ForCausalLM(config)

        # Load only the backbone weights, prefixed for Qwen2ForCausalLM ("model.").
        backbone_sd = self._load_backbone_state_dict(pretrained_model_name)
        prefixed = {f"model.{key}": tensor for key, tensor in backbone_sd.items()}
        missing, unexpected = model.load_state_dict(prefixed, strict=False)
        # Tied embeddings: lm_head shares embed_tokens, so it is expected missing.
        unexpected = [k for k in unexpected]
        missing = [k for k in missing if k != "lm_head.weight"]
        if unexpected:
            raise RuntimeError(f"Unexpected backbone keys: {unexpected[:5]} ...")
        if missing:
            raise RuntimeError(f"Missing backbone keys: {missing[:5]} ...")
        model.tie_weights()

        if dtype_override is not None:
            model = model.to(dtype_override)
        else:
            model = model.to(torch.bfloat16)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the VibeVoice backbone.

        Args:
            dtype_override: Unused for integer token inputs; kept for API parity.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        self._last_inputs = inputs
        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode the greedy next-token prediction into a human-readable string.

        Predicts at the last non-padding position so the result is meaningful
        even when inputs are right-padded to a fixed length.

        Args:
            outputs: Model output from a forward pass.
            inputs: Optional inputs dict; falls back to the last inputs produced
                    by ``load_inputs`` to locate the last real token.

        Returns:
            str: The greedily-predicted next token.
        """
        if self.tokenizer is None:
            self._load_tokenizer()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        inputs = inputs if inputs is not None else getattr(self, "_last_inputs", None)
        pos = -1
        if inputs is not None and "attention_mask" in inputs:
            real = int(inputs["attention_mask"][0].sum().item())
            pos = max(0, real - 1)
        next_id = int(torch.argmax(logits[0, pos, :]))
        return self.tokenizer.decode([next_id])
