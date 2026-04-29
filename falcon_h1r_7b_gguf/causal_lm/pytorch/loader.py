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

    Transformers 5.x has FalconH1ForCausalLM but lacks GGUF loading support
    for the falcon-h1 architecture. The following gaps need bridging:

    1. GGUF_CONFIG_MAPPING is missing the falcon-h1 → FalconH1Config field map.
       - attention.key_length maps to head_dim (FalconH1 does not satisfy
         head_dim = hidden_size / num_attention_heads).
       - ssm.time_step_rank maps to mamba_n_heads.
    2. GGUF_CONFIG_DEFAULTS_MAPPING needs mamba_rms_norm=True and
       mamba_norm_before_gate=False (not stored in GGUF metadata, but the
       trained model uses them).
    3. get_gguf_hf_weights_map doesn't remap model_type "falcon_h1" (HF,
       underscore) to "falcon-h1" (gguf-py, hyphen).
    4. GGUF_TO_FAST_CONVERTERS has no entry for falcon-h1.
    5. Tensor shape transforms and missing name mappings:
       - ssm_a GGUF shape [num_heads, 1] must be log(-x) then squeezed to [num_heads].
       - ssm_d GGUF shape [num_heads, 1] must be squeezed to [num_heads].
       - ssm_conv1d.weight GGUF shape [N, K] must gain a group dim → [N, 1, K].
       - blk.N.ssm_dt.bias (GGUF) → model.layers.N.mamba.dt_bias (HF).
       - blk.N.ffn_norm (GGUF, no .weight) → model.layers.N.pre_ff_layernorm.weight.
    """
    import re
    import numpy as np
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        GGUFTensor,
        TensorProcessor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "falcon-h1" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    # 1. Register config key mapping.
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["falcon-h1"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": None,       # stored in rope_parameters, not flat
        "rope.dimension_count": None,
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",  # falcon-h1 head_dim ≠ hidden/heads
        "attention.value_length": None,
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.state_size": "mamba_d_state",
        "ssm.time_step_rank": "mamba_n_heads",  # 24 = mamba_d_ssm / mamba_d_head
        "ssm.inner_size": "mamba_d_ssm",
        "ssm.group_count": "mamba_n_groups",
        "vocab_size": "vocab_size",
    }

    # GGUF_SUPPORTED_ARCHITECTURES is a snapshot list, update separately.
    GGUF_SUPPORTED_ARCHITECTURES.append("falcon-h1")

    # 2. Set config defaults not stored in GGUF metadata.
    from transformers.integrations.ggml import GGUF_CONFIG_DEFAULTS_MAPPING
    GGUF_CONFIG_DEFAULTS_MAPPING["falcon-h1"] = {
        "mamba_rms_norm": True,         # actual trained model uses True
        "mamba_norm_before_gate": False, # actual trained model uses False
    }

    # 3. Register GPT-2/BPE tokenizer converter (same family as falcon).
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )
    if "falcon-h1" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["falcon-h1"] = GGUFGPTConverter
    # tokenization_utils_tokenizers.py reads model_type (underscore), not the
    # raw GGUF arch string (hyphen), so register the post-rename key as well.
    if "falcon_h1" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["falcon_h1"] = GGUFGPTConverter

    # 4. Custom tensor processor for falcon-h1 shape fixups.
    _LAYER_RE = re.compile(r"blk\.(\d+)\.")

    class FalconH1TensorProcessor(TensorProcessor):
        def process(self, weights, name, **kwargs):
            if "ssm_conv1d.weight" in name:
                # GGUF shape [out_ch, K]; Conv1d expects [out_ch, 1, K].
                weights = np.expand_dims(weights, axis=1)
            elif name.endswith(".ssm_a"):
                # Convert raw A (stored as negative) to A_log = log(-A),
                # then squeeze the trailing 1 dim: [num_heads, 1] → [num_heads].
                # Use endswith to avoid matching ssm_a_scale or similar names.
                weights = np.log(-weights)
                weights = np.squeeze(weights, axis=-1)
            elif name.endswith(".ssm_d"):
                # Squeeze trailing 1 dim: [num_heads, 1] → [num_heads].
                # Use endswith to avoid matching ssm_dt.bias.
                weights = np.squeeze(weights, axis=-1)
            return GGUFTensor(weights, name, {})

        def perform_fallback_tensor_mapping(
            self, gguf_to_hf_name_map, suffix, qual_name, hf_name
        ):
            # dt_bias: HF "model.layers.N.mamba.dt_bias" has no matching gguf
            # name_map entry because gguf-py maps dt_proj → ssm_dt, not dt_bias.
            if hf_name.endswith(".mamba.dt_bias"):
                m = _LAYER_RE.search(hf_name)
                if m:
                    gguf_to_hf_name_map[f"blk.{m.group(1)}.ssm_dt.bias"] = (
                        qual_name + hf_name
                    )

    from transformers.modeling_gguf_pytorch_utils import TENSOR_PROCESSORS
    TENSOR_PROCESSORS["falcon-h1"] = FalconH1TensorProcessor

    # 5. Patch load_gguf_checkpoint to rename model_type "falcon-h1" →
    #    "falcon_h1" so AutoConfig resolves to FalconH1Config.
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

    # 6. Patch get_gguf_hf_weights_map to:
    #    (a) remap model_type "falcon_h1" (HF underscore) → "falcon-h1"
    #        (gguf-py hyphen), mirroring the cohere→command-r pattern, and
    #    (b) add the ffn_norm without .weight suffix so the GGUF tensor
    #        "blk.N.ffn_norm" (stored without .weight) maps to the HF param
    #        "model.layers.N.pre_ff_layernorm.weight".
    _orig_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            resolved = hf_model.config.model_type
        else:
            resolved = model_type
        if resolved == "falcon_h1":
            model_type = "falcon-h1"
        result = _orig_weights_map(
            hf_model, processor, model_type=model_type,
            num_layers=num_layers, qual_name=qual_name,
        )
        # GGUF stores ffn_norm without a .weight suffix; add the suffix-less
        # variant so the tensor is found during loading.
        for k in list(result.keys()):
            if k.endswith(".ffn_norm.weight"):
                bare = k[: -len(".weight")]
                result.setdefault(bare, result[k])
        return result

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
    if hasattr(modeling_utils, "get_gguf_hf_weights_map"):
        modeling_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


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
