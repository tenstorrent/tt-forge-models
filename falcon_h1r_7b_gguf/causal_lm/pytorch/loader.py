# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon H1R 7B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

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


def _patch_transformers_falcon_h1_gguf():
    """Monkey-patch transformers to add falcon-h1 GGUF architecture support.

    The Falcon H1R model uses the 'falcon-h1' architecture identifier in its
    GGUF metadata. Transformers has FalconH1ForCausalLM (model_type='falcon_h1')
    but lacks GGUF loading support for the 'falcon-h1' architecture key. We
    bridge the gap by registering the config mapping, remapping model_type, and
    registering a tokenizer converter for the 'falcon_h1' model type.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "falcon-h1" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_CONFIG_MAPPING["falcon-h1"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.state_size": "mamba_d_state",
        "ssm.group_count": "mamba_n_groups",
        "ssm.inner_size": None,
        "ssm.time_step_rank": None,
        "attention.key_length": None,
        "attention.value_length": None,
    }
    GGUF_SUPPORTED_ARCHITECTURES.append("falcon-h1")
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["falcon-h1"] = GGUF_CONFIG_MAPPING[
        "falcon-h1"
    ]

    # The tokenizer converter lookup uses model_type as the key after remapping,
    # so register 'falcon_h1' with the GPT-2 converter (tokenizer.ggml.model=gpt2).
    if "falcon_h1" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["falcon_h1"] = GGUFGPTConverter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "falcon-h1":
            config["model_type"] = "falcon_h1"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_falcon_h1_gguf()


class ModelVariant(StrEnum):
    """Available Falcon H1R 7B GGUF model variants for causal language modeling."""

    FALCON_H1R_7B_Q4_K_M = "Q4_K_M"
    FALCON_H1R_7B_TIIUAE_Q4_K_M = "tiiuae_Q4_K_M"


# Map variants to their GGUF filenames
_GGUF_FILES = {
    ModelVariant.FALCON_H1R_7B_Q4_K_M: "Falcon-H1R-7B.i1-Q4_K_M.gguf",
    ModelVariant.FALCON_H1R_7B_TIIUAE_Q4_K_M: "Falcon-H1R-7B-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Falcon H1R 7B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FALCON_H1R_7B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Falcon-H1R-7B-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.FALCON_H1R_7B_TIIUAE_Q4_K_M: LLMModelConfig(
            pretrained_model_name="tiiuae/Falcon-H1R-7B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON_H1R_7B_Q4_K_M

    sample_text = "What is your favorite city?"

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
            model="Falcon H1R 7B GGUF",
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
        tokenizer_kwargs["gguf_file"] = _GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = _GGUF_FILES[self._variant]

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=_GGUF_FILES[self._variant]
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=_GGUF_FILES[self._variant],
        )
        return self.config
