# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi-1.5 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _import_time_gguf_loader,
    get_gguf_hf_weights_map as _import_time_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

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

# phi2 is the GGUF architecture name for Phi-1.5/Phi-2 models.
# transformers 5.x does not register it; patch in the mapping so
# load_gguf_checkpoint can parse the checkpoint metadata.
_PHI2_CONFIG_KEYS = {
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "block_count": "num_hidden_layers",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_epsilon": "layer_norm_eps",
    "rope.dimension_count": None,
    "vocab_size": "vocab_size",
}


def _patch_phi2_support():
    if "phi2" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("phi2")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "phi2", _PHI2_CONFIG_KEYS
    )
    GGUF_TO_FAST_CONVERTERS.setdefault("phi2", GGUFGPTConverter)


def _find_true_gguf_loader(fn):
    """Traverse the loader patch chain to find transformers' load_gguf_checkpoint.

    Many loaders monkey-patch load_gguf_checkpoint with narrow signatures that
    don't forward model_to_load.  The true transformers original explicitly
    declares model_to_load as a parameter.  We search through both __globals__
    (for module-level orig captures) and __closure__ (for inner-function orig
    captures), until we find the function with model_to_load in its params.
    """
    seen = set()

    def _search(f):
        fn_id = id(f)
        if fn_id in seen:
            return None
        seen.add(fn_id)
        code = getattr(f, "__code__", None)
        if code is None:
            return None
        # Found: this is the true transformers load_gguf_checkpoint
        if "model_to_load" in code.co_varnames[: code.co_argcount]:
            return f
        # Search closure cells (captures from enclosing function scope)
        for cell in f.__closure__ or []:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val) and hasattr(val, "__code__"):
                result = _search(val)
                if result is not None:
                    return result
        # Search globals referenced by name in this function's bytecode
        g = getattr(f, "__globals__", {})
        for name in code.co_names:
            val = g.get(name)
            if val is not None and callable(val) and hasattr(val, "__code__"):
                result = _search(val)
                if result is not None:
                    return result
        return None

    return _search(fn)


def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_phi2_support()
    # _gguf_utils.load_gguf_checkpoint may have been clobbered by later loaders
    # with narrow signatures that don't forward model_to_load.  Bypass the patch
    # chain by starting the traversal from _import_time_gguf_loader — the value
    # of load_gguf_checkpoint when phi1_5 was first imported.  That may itself
    # be a narrow-sig patched function, but the closure chain from it eventually
    # leads to the true transformers original that accepts model_to_load.
    true_orig = _find_true_gguf_loader(_import_time_gguf_loader)
    if true_orig is None:
        # _import_time_gguf_loader is already the true original (no chain yet)
        true_orig = _import_time_gguf_loader
    result = true_orig(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "phi2":
        result["config"]["model_type"] = "phi"
    return result


def _patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
    # gguf-py MODEL_ARCH_NAMES maps PHI2->"phi2", not "phi"; remap so the arch
    # lookup succeeds.
    if model_type is None:
        model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
    if model_type == "phi":
        model_type = "phi2"
    return _import_time_weights_map(hf_model, processor, model_type, num_layers, qual_name)


def _apply_patches():
    _patch_phi2_support()
    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_apply_patches()


class ModelVariant(StrEnum):
    """Available Phi-1.5 GGUF model variants for causal language modeling."""

    PHI_1_5_Q4_K_M = "Phi_1_5_Q4_K_M"


class ModelLoader(ForgeModel):
    """Phi-1.5 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PHI_1_5_Q4_K_M: LLMModelConfig(
            pretrained_model_name="TKDKid1000/phi-1_5-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PHI_1_5_Q4_K_M

    GGUF_FILE = "phi-1_5-Q4_K_M.gguf"

    sample_text = "Africa is an emerging economy because"

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
            model="Phi-1.5 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        # Re-apply at call time: other loaders imported after us may have
        # clobbered _gguf_utils.load_gguf_checkpoint.
        _apply_patches()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            # GGUFGPTConverter doesn't propagate special tokens; fall back to
            # the phi-1.5 / GPT-2 end-of-text token which is always in vocab.
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        # Re-apply at call time: other loaders imported after us may have
        # clobbered _gguf_utils.load_gguf_checkpoint.
        _apply_patches()

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

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        _apply_patches()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
