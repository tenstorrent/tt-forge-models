# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon-H1-Tiny-R-90M GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_transformers_falcon_h1_gguf():
    """Monkey-patch transformers to add falcon-h1 GGUF architecture support.

    Transformers 5.x has FalconH1ForCausalLM but lacks GGUF loading support for
    the falcon-h1 architecture. We register the architecture, config key mapping,
    and a tensor processor that fixes shape mismatches in the Mamba tensors.
    """
    import numpy as np
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        TensorProcessor,
        GGUFTensor,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

    if "falcon-h1" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    class FalconH1TensorProcessor(TensorProcessor):
        """Handle Mamba-2 tensor shape quirks in falcon-h1 GGUF files.

        The GGUF exporter stores ssm_a and ssm_d as (num_heads, 1) and
        ssm_conv1d.weight as (kernel_size, channels), but the HF model
        expects (num_heads,) and (channels, 1, kernel_size) respectively.
        """

        def process(self, weights, name, **kwargs):
            if "ssm_conv1d.weight" in name:
                # (channels, kernel_size) -> (channels, 1, kernel_size)
                weights = np.expand_dims(weights, axis=1)
            elif "ssm_a" in name:
                # (num_heads, 1) -> (num_heads,) for A_log
                if weights.ndim == 2 and weights.shape[-1] == 1:
                    weights = weights.squeeze(-1)
                weights = np.log(-weights)
            elif "ssm_d" in name:
                # (num_heads, 1) -> (num_heads,) for D
                if weights.ndim == 2 and weights.shape[-1] == 1:
                    weights = weights.squeeze(-1)
            return GGUFTensor(weights, name, {})

    TENSOR_PROCESSORS["falcon-h1"] = FalconH1TensorProcessor

    GGUF_SUPPORTED_ARCHITECTURES.append("falcon-h1")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["falcon-h1"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": None,
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.inner_size": "mamba_d_ssm",
        "ssm.state_size": "mamba_d_state",
        "ssm.time_step_rank": "mamba_n_heads",
        "ssm.group_count": "mamba_n_groups",
    }

    if "falcon-h1" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["falcon-h1"] = GGUFGPTConverter
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

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "falcon_h1":
            model_type = "falcon-h1"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_falcon_h1_gguf()

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
    """Available Falcon-H1-Tiny-R-90M GGUF model variants for causal language modeling."""

    FALCON_H1_TINY_R_90M_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Falcon-H1-Tiny-R-90M GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FALCON_H1_TINY_R_90M_Q4_K_M: LLMModelConfig(
            pretrained_model_name="tiiuae/Falcon-H1-Tiny-R-90M-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON_H1_TINY_R_90M_Q4_K_M

    GGUF_FILE = "Falcon-H1R-Tiny-90M-Q4_K_M.gguf"

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
            model="Falcon-H1-Tiny-R-90M GGUF",
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
        # GGUFGPTConverter adds <s>/<\/s> tokens beyond vocab_size (IDs 32768/32769).
        # Override special tokens to use in-vocab IDs from the GGUF metadata:
        # token 0=<|pad|>, token 11=<|end_of_text|>, token 17=<|begin_of_text|>.
        self.tokenizer.bos_token_id = 17
        self.tokenizer.eos_token_id = 11
        self.tokenizer.pad_token_id = 0

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
