# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llasa (Llama3-8B TTS) model loader implementation.

Llasa-8B is a text-to-speech model that is initialized from Llama 3.1 8B and
fine-tuned with an expanded vocabulary (text tokens + XCodec2 speech tokens,
vocab 193800). Architecturally it is a plain ``LlamaForCausalLM``: given a text
prompt wrapped in Llasa's TTS chat format, it autoregressively predicts speech
tokens which an external XCodec2 decoder turns into a waveform.

Because the model is a standard causal LM, the entire on-device graph is just the
language-model forward (static shapes) -- there is no multimodal merge or
dynamic op to special-case. The XCodec2 audio decoder is not part of this graph;
this loader covers the LM forward (the part that runs on Tenstorrent hardware).
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional
import torch

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
    """Available Llasa model variants."""

    LLASA_8B = "Llasa-8B"


class ModelLoader(ForgeModel):
    """Llasa (Llama3-8B TTS) loader implementation for causal LM TTS tasks."""

    _VARIANTS = {
        ModelVariant.LLASA_8B: LLMModelConfig(
            pretrained_model_name="HKUSTAudio/Llasa-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLASA_8B

    # Sample text to be synthesized into speech.
    sample_text = "Tenstorrent builds hardware and software for AI."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Llasa",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Llasa model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                dtype. Bring-up uses torch.bfloat16.

        Returns:
            torch.nn.Module: The Llasa (LlamaForCausalLM) model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config

        return model

    def _build_tts_prompt(self, text: str):
        """Build the Llasa TTS chat prompt for ``text``.

        Mirrors Llasa's documented usage: the text to synthesize is wrapped in
        ``<|TEXT_UNDERSTANDING_START|>...<|TEXT_UNDERSTANDING_END|>`` and the
        assistant turn is primed with ``<|SPEECH_GENERATION_START|>`` so the
        model continues by emitting speech tokens.
        """
        return [
            {
                "role": "user",
                "content": (
                    "Convert the text to speech:"
                    f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
                ),
            },
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Build static-shape inputs for the Llasa causal LM TTS forward.

        Extra runner kwargs (e.g. ``run_phase``, ``seq_len``) are accepted and
        ignored.

        Args:
            dtype_override: Optional torch.dtype (floating inputs only; ids are
                left as int).
            batch_size: Batch size (replicated along dim 0). ``None`` -> 1.

        Returns:
            dict: ``{"input_ids": Tensor, "attention_mask": Tensor}``.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        batch_size = batch_size or 1

        encoded = self.tokenizer.apply_chat_template(
            self._build_tts_prompt(self.sample_text),
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        )
        # apply_chat_template may return a bare tensor or a BatchEncoding/dict
        # depending on the transformers version.
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad to a static length so the on-device graph has fixed shapes.
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def decode_output(self, outputs, dtype_override=None):
        """Greedy-decode the next speech token from logits for a quick check."""
        if self.tokenizer is None:
            self._load_tokenizer()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        next_id = logits[:, self.seq_len - 1, :].argmax(dim=-1)
        return self.tokenizer.decode(next_id.tolist())

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style tensor-parallel shard spec for the Llasa LM.

        Column-parallel on q/k/v/gate/up, row-parallel on o/down. Used only for
        multichip bring-up; single-device runs ignore this.
        """
        shard_specs = {}
        shard_specs[model.lm_head.weight] = ("model", "batch")
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Llasa model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
