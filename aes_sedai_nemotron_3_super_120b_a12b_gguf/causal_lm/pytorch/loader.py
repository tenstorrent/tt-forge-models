# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling.
"""
from typing import Optional

import torch
from transformers import AutoTokenizer

# Patch load_gguf_checkpoint to recognise the nemotron_h_moe GGUF architecture.
# The transformers GGUF pipeline does not support loading NemotronH weights from
# GGUF, so we only need the patch to allow tokeniser loading to succeed; the
# config and model are constructed directly below.
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.configuration_utils as _config_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
)


def _patch_nemotron_h_moe_support():
    """Add nemotron_h_moe as an alias in the GGUF architecture registry.

    The nemotron_h_moe GGUF architecture (NVIDIA Nemotron-H hybrid MoE) is not
    recognised by any published version of transformers.  We register it here
    so that the GGUF tokeniser loader can parse the file without crashing; the
    actual config and model weights are loaded via a different path.
    """
    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")
    for section in GGUF_TO_TRANSFORMERS_MAPPING:
        if "nemotron" in GGUF_TO_TRANSFORMERS_MAPPING[section]:
            GGUF_TO_TRANSFORMERS_MAPPING[section][
                "nemotron_h_moe"
            ] = GGUF_TO_TRANSFORMERS_MAPPING[section]["nemotron"]
    try:
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

        if "nemotron" in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUF_TO_FAST_CONVERTERS[
                "nemotron"
            ]
    except ImportError:
        pass
    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "nemotron" in _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING[
                "nemotron_h_moe"
            ] = _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["nemotron"]


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "nemotron":
        result["config"]["model_type"] = "nemotron_h"
    return result


_patch_nemotron_h_moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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

# Full hybrid-override pattern for NVIDIA-Nemotron-3-Super-120B-A12B-BF16.
# M = mamba, E = moe, * = attention  (88 layers total)
_FULL_PATTERN = (
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
    "EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
)


def _build_nemotron_h_config(num_layers: Optional[int] = None):
    """Construct a NemotronHConfig from the known model parameters.

    Because transformers has no GGUF weight-loading pipeline for nemotron_h_moe,
    we build the config directly from the published nvidia model parameters rather
    than parsing the GGUF file.  num_layers truncates the layer pattern so the
    model fits into available memory during compile-only testing.
    """
    from transformers import NemotronHConfig

    pattern = _FULL_PATTERN
    if num_layers is not None:
        pattern = pattern[:num_layers]

    return NemotronHConfig(
        vocab_size=131072,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=262144,
        intermediate_size=2688,
        ssm_state_size=128,
        n_groups=8,
        expand=2,
        conv_kernel=4,
        mamba_num_heads=128,
        mamba_head_dim=64,
        n_routed_experts=512,
        num_experts_per_tok=22,
        moe_intermediate_size=2688,
        moe_shared_expert_intermediate_size=5376,
        moe_latent_size=1024,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
        moe_shared_expert_overlap=False,
        layer_norm_epsilon=1e-05,
        rope_theta=10000,
        use_mamba_kernels=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        hybrid_override_pattern=pattern,
        num_nextn_predict_layers=0,
    )


class ModelVariant(StrEnum):
    """Available AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model variants for causal language modeling."""

    AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF = "AesSedai_3_Super_120B_A12B_GGUF"


class ModelLoader(ForgeModel):
    """AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF

    GGUF_FILE = (
        "Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M-00001-of-00003.gguf"
    )

    # Default to 2 layers for compile-only testing: full 88-layer model exceeds
    # practical memory limits when instantiated with random weights.
    DEFAULT_NUM_LAYERS = 2

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = (
            num_layers if num_layers is not None else self.DEFAULT_NUM_LAYERS
        )

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF",
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

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if self.config is None:
            self.load_config()

        from transformers import NemotronHForCausalLM

        model = NemotronHForCausalLM(self.config).eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_config(self):
        self.config = _build_nemotron_h_config(num_layers=self.num_layers)
        return self.config
