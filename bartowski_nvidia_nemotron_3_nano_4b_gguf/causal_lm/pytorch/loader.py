# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski nvidia Nemotron-3-Nano-4B GGUF model loader implementation for causal language modeling.
"""
import re

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_transformers_nemotron_h_gguf():
    """Register nemotron_h as a supported GGUF architecture.

    transformers 5.6.2 has NemotronHForCausalLM but no GGUF loading support.
    This patch adds the config mapping, a tensor processor, and derives the
    hybrid layer pattern from the GGUF tensor names.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        GGUFTensor,
        TensorProcessor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "nemotron_h" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["nemotron_h"] = {
        "context_length": "max_position_embeddings",
        "block_count": None,  # derived from layers_block_type
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": None,  # GGUF stores 0; derived from tensors
        "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
        "attention.key_length": "head_dim",
        "vocab_size": "vocab_size",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.state_size": "ssm_state_size",
        "ssm.group_count": "mamba_n_groups",
        "ssm.time_step_rank": "mamba_num_heads",
        "ssm.inner_size": None,  # used below to derive mamba_head_dim
        "rope.dimension_count": None,
        "attention.value_length": None,
        "rope.scaling.finetuned": None,
        "attention.layer_norm_epsilon": None,
    }

    class NemotronHTensorProcessor(TensorProcessor):
        def preprocess_name(self, hf_name: str) -> str:
            # gguf-py maps backbone.layers.N.* but NemotronH uses model.layers.N.*
            hf_name = re.sub(
                r"^model\.layers\.(\d+)\.", r"backbone.layers.\1.", hf_name
            )
            return hf_name

        def perform_fallback_tensor_mapping(
            self, gguf_to_hf_name_map, suffix, qual_name, hf_name
        ):
            # dt_bias in HF maps to blk.N.ssm_dt.bias in GGUF
            m = re.match(r"backbone\.layers\.(\d+)\.mixer\.dt_bias$", hf_name)
            if m:
                gguf_name = f"blk.{m.group(1)}.ssm_dt.bias"
                gguf_to_hf_name_map[gguf_name] = qual_name + re.sub(
                    r"^backbone", "model", hf_name
                )

        def process(self, weights, name, **kwargs):
            if "ssm_conv1d.weight" in name:
                # Dequantized shape is [conv_dim, kernel]; HF expects [conv_dim, 1, kernel]
                weights = np.expand_dims(weights, axis=1)
            elif "ssm_a" in name:
                # GGUF stores raw negative A; A_log = log(-A), flatten to 1D
                weights = np.log(-weights).flatten()
            elif "ssm_d" in name:
                # GGUF: possibly [1, H] or [H, 1] -> HF: [H]
                weights = weights.flatten()
            elif "ssm_norm.weight" in name:
                # GGUF: [group_size, n_groups] -> HF: [intermediate_size]
                weights = weights.flatten()
            return GGUFTensor(weights, name, {})

    TENSOR_PROCESSORS["nemotron_h"] = NemotronHTensorProcessor

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})

        if config.get("model_type") != "nemotron_h":
            return result

        from gguf import GGUFReader
        from transformers import NemotronHConfig

        gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
        if not gguf_path:
            return result

        reader = GGUFReader(gguf_path)
        tensor_names = {t.name for t in reader.tensors}

        # Derive layers_block_type from tensor presence
        block_count = 0
        for t in reader.tensors:
            m = re.match(r"blk\.(\d+)\.", t.name)
            if m:
                block_count = max(block_count, int(m.group(1)) + 1)

        layer_types = []
        for i in range(block_count):
            if f"blk.{i}.ssm_a" in tensor_names:
                layer_types.append("mamba")
            elif f"blk.{i}.attn_q" in tensor_names:
                layer_types.append("attention")
            else:
                layer_types.append("mlp")

        config["hybrid_override_pattern"] = NemotronHConfig._list_to_pattern(
            layer_types
        )

        # Derive mamba_head_dim from ssm.inner_size / mamba_num_heads
        mamba_num_heads = config.get("mamba_num_heads", 0)
        # Read ssm.inner_size directly from GGUF
        if "nemotron_h.ssm.inner_size" in reader.fields:
            inner_size = int(reader.fields["nemotron_h.ssm.inner_size"].parts[-1][0])
            if mamba_num_heads:
                config["mamba_head_dim"] = inner_size // mamba_num_heads

        # intermediate_size is stored per-layer in GGUF (0 for mamba, N for mlp/attn)
        if isinstance(config.get("intermediate_size"), list):
            config["intermediate_size"] = max(config["intermediate_size"])

        # Derive num_key_value_heads from attn_k tensor shape (GGUF stores 0)
        head_dim = config.get("head_dim", 128)
        for t in reader.tensors:
            m = re.match(r"blk\.(\d+)\.attn_k\.weight", t.name)
            if m:
                config["num_key_value_heads"] = int(t.shape[1]) // head_dim
                break

        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.models.auto.tokenization_auto as tok_auto

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Register nemotron_h in fast tokenizer converters if nemotron is present
    try:
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

        if "nemotron" in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["nemotron_h"] = GGUF_TO_FAST_CONVERTERS["nemotron"]
    except (ImportError, AttributeError):
        pass


_patch_transformers_nemotron_h_gguf()


class ModelVariant(StrEnum):
    """Available bartowski nvidia Nemotron-3-Nano-4B GGUF model variants for causal language modeling."""

    BARTOWSKI_NVIDIA_NEMOTRON_3_NANO_4B_Q4_K_M_GGUF = (
        "nvidia_Nemotron_3_Nano_4B_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """bartowski nvidia Nemotron-3-Nano-4B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_NVIDIA_NEMOTRON_3_NANO_4B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/nvidia_Nemotron-3-Nano-4B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_NVIDIA_NEMOTRON_3_NANO_4B_Q4_K_M_GGUF

    GGUF_FILE = "nvidia_Nemotron-3-Nano-4B-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="bartowski nvidia Nemotron-3-Nano-4B GGUF",
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
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
