# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Darkmere 14B v0.1 i1 GGUF model loader implementation for causal language modeling.
"""
import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

# --------------------------------------------------------------------------- #
# Patch: register the 'mistral3' GGUF architecture.
#
# The GGUF file declares general.architecture = 'mistral3' (Ministral 3 14B).
# Transformers 5.x knows 'mistral3' as a VLM only; there is no GGUF
# config-mapping entry for it.  We:
#   1. Register it in GGUF tables at import time (persists across all loaders).
#   2. In load_model/load_config, temporarily wrap the current outermost
#      load_gguf_checkpoint to strip 'model_to_load' before forwarding to the
#      old-signature chain, and inject a None-safe get_gguf_hf_weights_map
#      that restores the model reference at the TRUE ORIGINAL call site.
# --------------------------------------------------------------------------- #

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import (
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter


def _patch_mistral3_support():
    """Register mistral3 GGUF architecture for Ministral 3 text models.

    Mistral3 in gguf-py refers to the text-only Ministral 3 architecture.
    Transformers 5.x 'mistral3' is a VLM so we remap model_type to
    'ministral3' (Ministral3ForCausalLM) after loading the GGUF config.
    """
    if "mistral3" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "mistral3",
        {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": "head_dim",
            "vocab_size": "vocab_size",
        },
    )
    # Ministral3 uses BPE tokenizer identical to llama
    GGUF_TO_FAST_CONVERTERS.setdefault("mistral3", GGUFLlamaConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("ministral3", GGUFLlamaConverter)


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name="", **kwargs
):
    """Remap ministral3 → mistral3 for gguf-py MODEL_ARCH_NAMES lookup."""
    if model_type is None and hf_model is not None:
        model_type = hf_model.config.model_type
    if model_type == "ministral3":
        model_type = "mistral3"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type=model_type, num_layers=num_layers,
        qual_name=qual_name, **kwargs
    )


# Register at import time (persists even when other loaders overwrite load_gguf_checkpoint)
_patch_mistral3_support()
# Patch get_gguf_hf_weights_map (fewer loaders overwrite this; any that do chain through ours)
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

# --------------------------------------------------------------------------- #

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
    """Available Darkmere 14B v0.1 i1 GGUF model variants for causal language modeling."""

    DARKMERE_14B_V01_I1_GGUF = "14B_v0.1_i1_GGUF"


class ModelLoader(ForgeModel):
    """Darkmere 14B v0.1 i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DARKMERE_14B_V01_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Darkmere-14B-v0.1-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DARKMERE_14B_V01_I1_GGUF

    GGUF_FILE = "Darkmere-14B-v0.1.i1-Q4_K_M.gguf"

    sample_text = "Write a Python function that checks if a number is prime."

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
            model="Darkmere 14B v0.1 i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @contextlib.contextmanager
    def _mistral3_load_context(self):
        """Temporarily wrap load_gguf_checkpoint to handle model_to_load.

        Many loaders in the test suite patch load_gguf_checkpoint with a
        2-argument signature that does not accept model_to_load.  Transformers
        5.x always passes model_to_load=<dummy_model> for GGUF tensor loading.
        We work around this by:
          a) Making the outermost function accept *args/**kwargs.
          b) Stripping model_to_load before forwarding to the old-signature chain.
          c) Temporarily making get_gguf_hf_weights_map None-safe so the TRUE
             ORIGINAL can still build the tensor-name map using the saved model.
        """
        _outer_load = _gguf_utils.load_gguf_checkpoint
        _outer_get_map = _gguf_utils.get_gguf_hf_weights_map
        _holder = [None]  # [model_to_load]

        def _temp_load(*args, **kwargs):
            _patch_mistral3_support()
            model_to_load = kwargs.pop("model_to_load", None)
            if model_to_load is not None:
                _holder[0] = model_to_load
            return _outer_load(*args, **kwargs)

        def _temp_get_map(hf_model, processor, model_type=None, num_layers=None, qual_name="", **kw):
            # When the old-sig chain passes model_to_load=None to the true
            # original, recover the real model from the holder.
            if hf_model is None and _holder[0] is not None:
                hf_model = _holder[0]
            return _outer_get_map(
                hf_model, processor, model_type=model_type, num_layers=num_layers,
                qual_name=qual_name, **kw
            )

        _gguf_utils.load_gguf_checkpoint = _temp_load
        _gguf_utils.get_gguf_hf_weights_map = _temp_get_map
        try:
            yield
        finally:
            _gguf_utils.load_gguf_checkpoint = _outer_load
            _gguf_utils.get_gguf_hf_weights_map = _outer_get_map

    def _build_ministral3_config(self, pretrained_model_name, dtype_override=None):
        """Build Ministral3Config from GGUF metadata via load_gguf_checkpoint."""
        from huggingface_hub import hf_hub_download
        from transformers.models.ministral3 import Ministral3Config

        gguf_path = hf_hub_download(
            repo_id=pretrained_model_name, filename=self.GGUF_FILE
        )
        # Use current load_gguf_checkpoint (old-sig is fine for return_tensors=False)
        gguf_data = _gguf_utils.load_gguf_checkpoint(gguf_path, return_tensors=False)
        cfg = gguf_data.get("config", {})

        # Build rope_parameters from the parsed rope_theta
        rope_theta = cfg.pop("rope_theta", 1_000_000.0)
        rope_parameters = {
            "type": "yarn",
            "rope_theta": rope_theta,
            "factor": 16.0,
            "original_max_position_embeddings": 16384,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale_all_dim": 1.0,
            "mscale": 1.0,
            "llama_4_scaling_beta": 0.1,
        }

        # Extract known Ministral3Config fields and pass the rest as extras
        known = {
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps",
            "max_position_embeddings", "sliding_window",
        }
        init_kwargs = {k: v for k, v in cfg.items() if k in known}
        init_kwargs["rope_parameters"] = rope_parameters
        if dtype_override is not None:
            init_kwargs["torch_dtype"] = dtype_override

        return Ministral3Config(**init_kwargs)

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

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

        # Build explicit Ministral3Config to bypass GGUF-based config loading
        # (avoids AutoConfig resolving to the VLM Mistral3ForConditionalGeneration)
        config = self._build_ministral3_config(pretrained_model_name, dtype_override)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model_kwargs = {"config": config, "gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        with self._mistral3_load_context():
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

        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": self.sample_text}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = self.sample_text

        inputs = self.tokenizer(
            [text],
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

    def load_shard_spec(self, model):
        shard_specs = {}
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
        config = self._build_ministral3_config(
            self._variant_config.pretrained_model_name
        )
        self.config = config
        return config
