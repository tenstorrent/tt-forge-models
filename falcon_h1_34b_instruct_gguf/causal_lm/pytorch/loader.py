# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon-H1-34B-Instruct GGUF model loader implementation for causal language modeling.
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

    Transformers 5.x has FalconH1ForCausalLM but lacks GGUF loading support
    for the falcon-h1 architecture. This patch bridges the gap by:
    1. Registering falcon-h1 as a supported GGUF architecture.
    2. Adding the config/tokenizer field mappings.
    3. Fixing the model_type lookup in get_gguf_hf_weights_map so it resolves
       "falcon_h1" (HF name with underscore) to "falcon-h1" (gguf-py name with hyphen).
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "falcon-h1" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register falcon-h1 as a supported GGUF architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("falcon-h1")

    # 2. Add config field mapping for falcon-h1
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["falcon-h1"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": None,  # handled via rope_parameters in post-patch
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": None,  # derived
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.inner_size": "mamba_d_ssm",
        "ssm.state_size": "mamba_d_state",
        "ssm.time_step_rank": "mamba_n_heads",
        "ssm.group_count": "mamba_n_groups",
    }

    # 3. Register falcon-h1 tokenizer converter (GPT-2 BPE, same as falcon)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )

    if "falcon-h1" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["falcon-h1"] = GGUFGPTConverter
    # Tokenizer lookup uses model_type ("falcon_h1") not GGUF arch ("falcon-h1")
    if "falcon_h1" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["falcon_h1"] = GGUFGPTConverter

    # 4. Patch get_gguf_hf_weights_map to map "falcon_h1" (HF) -> "falcon-h1" (gguf-py)
    _orig_get_gguf_hf_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "falcon_h1":
            model_type = "falcon-h1"
        return _orig_get_gguf_hf_weights_map(
            hf_model, processor, model_type, num_layers, qual_name
        )

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

    # 5. Patch load_gguf_checkpoint to fix the config after GGUF parsing
    _orig_load_gguf_checkpoint = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load_gguf_checkpoint(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "falcon-h1":
            # Translate model_type to HF convention and set rope_parameters
            config["model_type"] = "falcon_h1"
            config["architectures"] = ["FalconH1ForCausalLM"]
            rope_freq = config.pop("rope_theta", 1e11)
            if "rope_parameters" not in config:
                config["rope_parameters"] = {
                    "rope_theta": rope_freq,
                    "rope_type": "default",
                }
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Patch any modules that imported load_gguf_checkpoint directly
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "load_gguf_checkpoint"):
        modeling_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


# Apply the monkey-patch at import time
_patch_transformers_falcon_h1_gguf()


class ModelVariant(StrEnum):
    """Available Falcon-H1-34B-Instruct GGUF model variants for causal language modeling."""

    FALCON_H1_34B_INSTRUCT_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Falcon-H1-34B-Instruct GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FALCON_H1_34B_INSTRUCT_Q4_K_M: LLMModelConfig(
            pretrained_model_name="tiiuae/Falcon-H1-34B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON_H1_34B_INSTRUCT_Q4_K_M

    GGUF_FILE = "Falcon-H1-34B-Instruct-Q4_K_M.gguf"

    # Non-GGUF repo for loading the full config (multipliers, rope_parameters, etc.)
    BASE_MODEL_NAME = "tiiuae/Falcon-H1-34B-Instruct"

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
            model="Falcon-H1-34B-Instruct GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        # Load tokenizer from the non-GGUF base model to avoid GGUF converter
        # issues that can produce out-of-range token IDs.
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _load_config(self):
        # Load the full config from the non-GGUF base model to get all parameters
        # (multipliers, rope_parameters, mamba_n_heads, etc.) that are not stored
        # in the GGUF file.
        config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        return config

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self._load_config()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs["config"] = config
        model_kwargs["ignore_mismatched_sizes"] = True

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
        self.config = self._load_config()
        return self.config
