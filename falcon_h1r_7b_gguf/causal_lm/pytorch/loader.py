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
    """Register falcon-h1 GGUF architecture support in global transformers tables.

    The Falcon H1R model uses the 'falcon-h1' architecture identifier in its
    GGUF metadata. Transformers has FalconH1ForCausalLM but lacks GGUF loading
    support for the 'falcon-h1' architecture key. We add all required entries
    globally so that the real load_gguf_checkpoint (restored at load time) can
    process the file correctly without any model_type remapping.
    """
    from transformers import FalconH1Config
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

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

    # Tokenizer converter lookup uses the architecture key directly.
    if "falcon-h1" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["falcon-h1"] = GGUFGPTConverter

    # AutoConfig.for_model / AutoTokenizer need 'falcon-h1' → FalconH1Config.
    CONFIG_MAPPING.register("falcon-h1", FalconH1Config, exist_ok=True)


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

    @staticmethod
    def _restore_real_load_gguf_checkpoint():
        """Restore the real load_gguf_checkpoint before calling from_pretrained.

        Other loaders in this repo patch load_gguf_checkpoint with restricted
        signatures lacking the model_to_load parameter. Since our global table
        registrations (GGUF_SUPPORTED_ARCHITECTURES, CONFIG_MAPPING, etc.) are
        already in place, restoring the real function is sufficient for correct
        GGUF loading without any model_type remapping.
        """
        import inspect
        import sys

        import transformers.modeling_gguf_pytorch_utils as _gguf_utils

        _current = _gguf_utils.load_gguf_checkpoint
        if "model_to_load" not in inspect.signature(_current).parameters:
            for _mod in list(sys.modules.values()):
                _candidate = getattr(_mod, "_orig_load_gguf_checkpoint", None)
                if _candidate is None:
                    continue
                try:
                    if "model_to_load" in inspect.signature(_candidate).parameters:
                        _gguf_utils.load_gguf_checkpoint = _candidate
                        break
                except Exception:
                    pass

    def _load_tokenizer(self, dtype_override=None):
        self._restore_real_load_gguf_checkpoint()

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

        self._restore_real_load_gguf_checkpoint()

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
