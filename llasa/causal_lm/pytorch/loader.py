# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llasa model loader implementation for text-to-speech.

Llasa (https://huggingface.co/HKUSTAudio/Llasa-8B) is a text-to-speech model
built on a ``LlamaForCausalLM`` backbone (fine-tuned from Llama-3.1-8B-Instruct).
The vocabulary is extended with discrete speech tokens (XCodec2 codes), so the
model autoregressively predicts speech tokens from a text prompt. For bringup we
exercise the standard single causal-LM forward pass over the speech-generation
prompt; the downstream XCodec2 decoder that turns speech tokens into a waveform
is a separate component and out of scope here.
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

    LLASA_8B = "8b"


class ModelLoader(ForgeModel):
    """Llasa model loader implementation for text-to-speech (causal LM backbone)."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LLASA_8B: LLMModelConfig(
            pretrained_model_name="HKUSTAudio/Llasa-8B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLASA_8B

    # Sample text to be synthesized into speech.
    sample_text = "Tenstorrent builds hardware for the future of artificial intelligence."

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
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Llasa",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Use the model's end-of-turn token for padding.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _build_prompt_ids(self):
        """Tokenize the Llasa speech-generation prompt for ``sample_text``.

        Mirrors the input format documented on the model card: the text is wrapped
        in the text-understanding markers and the assistant turn is primed with the
        speech-generation start token so the model continues with speech tokens.

        Returns:
            torch.Tensor: input_ids of shape (1, prompt_len).
        """
        formatted_text = (
            f"<|TEXT_UNDERSTANDING_START|>{self.sample_text}<|TEXT_UNDERSTANDING_END|>"
        )
        chat = [
            {
                "role": "user",
                "content": "Convert the text to speech:" + formatted_text,
            },
            {
                "role": "assistant",
                "content": "<|SPEECH_GENERATION_START|>",
            },
        ]
        encoded = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        )
        # Depending on the transformers version this returns either a tensor or a
        # BatchEncoding/dict; normalize to the input_ids tensor.
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids
        if isinstance(encoded, dict):
            return encoded["input_ids"]
        return encoded

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Llasa model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model loads in its native dtype (bfloat16).

        Returns:
            torch.nn.Module: The Llasa model instance for causal LM.
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

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llasa model.

        Args:
            dtype_override: Optional torch.dtype to override the default dtype of input tensors.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) suitable for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        input_ids = self._build_prompt_ids()
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask to a static length for compilation.
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def decode_output(self, max_new_tokens, model, inputs, tokenizer):
        """Greedily generate speech tokens and decode them to their token strings.

        For Llasa the generated ids are discrete speech tokens (``<|s_N|>``); turning
        them into an actual waveform requires the separate XCodec2 decoder, which is
        out of scope for this loader.

        Args:
            max_new_tokens (int): Maximum number of new (speech) tokens to generate.
            model (torch.nn.Module): The Llasa model.
            inputs (list): [input_ids, attention_mask] tensors.
            tokenizer: The tokenizer used to decode token ids.
        """
        current_pos = self.seq_len

        for _ in range(max_new_tokens):
            logits = model(*inputs)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            next_token_logits = logits[:, current_pos - 1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            inputs[0][:, current_pos] = next_token_id
            inputs[1][:, current_pos] = 1

            current_pos += 1

        valid_tokens = inputs[0][:, self.seq_len : current_pos].view(-1).tolist()
        answer = tokenizer.decode(valid_tokens, skip_special_tokens=False)
        return answer

    def get_mesh_config(self, num_devices: int):
        """Mesh shape for tensor-parallel sharding across ``num_devices`` chips."""
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron column->row tensor-parallel shard spec on a ("batch", "model") mesh."""
        shard_specs = {}
        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        shard_specs[model.model.norm.weight] = ("batch",)
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Llasa model variant.

        Returns:
            The configuration object for the Llasa model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
