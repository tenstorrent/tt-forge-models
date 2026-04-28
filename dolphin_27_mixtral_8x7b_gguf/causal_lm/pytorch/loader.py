# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dolphin 2.7 Mixtral 8x7B GGUF model loader implementation for causal language modeling.
"""
import glob
import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MixtralConfig

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
    """Available Dolphin 2.7 Mixtral 8x7B GGUF model variants for causal language modeling."""

    DOLPHIN_27_MIXTRAL_8X7B_GGUF = "2.7_Mixtral_8x7B_GGUF"


class ModelLoader(ForgeModel):
    """Dolphin 2.7 Mixtral 8x7B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DOLPHIN_27_MIXTRAL_8X7B_GGUF: LLMModelConfig(
            pretrained_model_name="TheBloke/dolphin-2.7-mixtral-8x7b-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOLPHIN_27_MIXTRAL_8X7B_GGUF

    GGUF_FILE = "dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf"

    sample_text = "What is the meaning of life?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Dolphin 2.7 Mixtral 8x7B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _build_mixtral_config_from_gguf(self):
        # transformers 5.x does not support loading Mixtral GGUF files whose
        # general.architecture is "llama" (the old llama.cpp convention for
        # Mixtral). AutoConfig.from_pretrained returns LlamaConfig, which
        # silently drops the expert fields and produces garbage output.
        # Work around by reading the GGUF metadata directly and constructing
        # a MixtralConfig with the correct parameters.
        try:
            from gguf import GGUFReader
            from transformers.modeling_gguf_pytorch_utils import _gguf_parse_value

            safe_id = self._variant_config.pretrained_model_name.replace("/", "--")
            pattern = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{safe_id}/snapshots/**/{self.GGUF_FILE}"
            )
            matches = glob.glob(pattern, recursive=True)
            if not matches:
                return None
            reader = GGUFReader(matches[0])

            def _read(field_name):
                f = reader.fields.get(field_name)
                if f is None:
                    return None
                return _gguf_parse_value(f.parts[f.data[0]], f.types)

            expert_count = _read("llama.expert_count")
            if not expert_count or int(expert_count) == 0:
                return None  # Not a MoE model; let AutoConfig handle it

            config = MixtralConfig(
                hidden_size=int(_read("llama.embedding_length") or 4096),
                intermediate_size=int(_read("llama.feed_forward_length") or 14336),
                num_hidden_layers=int(_read("llama.block_count") or 32),
                num_attention_heads=int(_read("llama.attention.head_count") or 32),
                num_key_value_heads=int(_read("llama.attention.head_count_kv") or 8),
                num_local_experts=int(expert_count),
                num_experts_per_tok=int(_read("llama.expert_used_count") or 2),
                rms_norm_eps=float(_read("llama.attention.layer_norm_rms_epsilon") or 1e-5),
                rope_theta=float(_read("llama.rope.freq_base") or 1000000.0),
                vocab_size=32000,
            )
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            return config
        except Exception:
            return None

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        # Build a MixtralConfig from GGUF metadata so the model loads with the
        # correct MoE architecture instead of being misidentified as LlamaForCausalLM.
        mixtral_config = self._build_mixtral_config_from_gguf()
        if mixtral_config is not None:
            model_kwargs["config"] = mixtral_config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # MixtralExperts.forward uses a Python for-loop whose trip count is
        # determined at runtime by nonzero(expert_hit). XLA cannot trace a
        # loop with a dynamic trip count, so switch to batched_mm which uses
        # only static tensor operations (einsum, scatter).
        model.config._experts_implementation = "batched_mm"

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        mixtral_config = self._build_mixtral_config_from_gguf()
        if mixtral_config is not None:
            self.config = mixtral_config
        else:
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
