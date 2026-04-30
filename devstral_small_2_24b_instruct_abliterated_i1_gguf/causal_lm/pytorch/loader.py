# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Devstral Small 2 24B Instruct Abliterated i1 GGUF model loader implementation for causal language modeling.
"""
import torch
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    get_gguf_hf_weights_map as _real_get_gguf_hf_weights_map,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter

# Grab the REAL transformers function before any further monkey-patching can
# occur.  Walk the patch chain by inspecting each wrapper's __globals__ for
# the "_orig_load_gguf_checkpoint" or similar name they captured.
def _find_real_gguf_fn():
    """Return the real transformers load_gguf_checkpoint, unwrapping tt_forge_models patches."""
    import types
    fn = _gguf_utils.__dict__["load_gguf_checkpoint"]
    seen = set()
    _ORIG_NAMES = (
        "_orig_load_gguf_checkpoint",
        "_real_orig_load_gguf_checkpoint",
        "orig_load",
        "_real_fn",
    )
    while True:
        mod = getattr(fn, "__module__", "") or ""
        if "transformers" in mod and "tt_forge_models" not in mod:
            return fn
        fn_id = id(fn)
        if fn_id in seen:
            # Cycle or stuck; return what we have
            return fn
        seen.add(fn_id)
        # Look in the function's global namespace for any saved "original"
        globs = getattr(fn, "__globals__", {})
        found = None
        for name in _ORIG_NAMES:
            val = globs.get(name)
            if callable(val) and isinstance(val, types.FunctionType):
                found = val
                break
        # Also check closures
        if found is None and getattr(fn, "__closure__", None):
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                    if isinstance(val, types.FunctionType):
                        found = val
                        break
                except ValueError:
                    pass
        if found is None:
            return fn
        fn = found


_REAL_LOAD_GGUF_CHECKPOINT = _find_real_gguf_fn()


def _patch_mistral3_data_structures():
    """Register mistral3 in GGUF data-structure mappings.

    Devstral Small 2 24B uses the GGUF architecture tag 'mistral3'
    (Mistral Small 3.x text backbone) which transformers 5.x does not yet
    recognise as a causal-LM target.  The parameter layout is identical to
    the existing 'mistral' mapping, so we simply alias it.  The tokenizer
    also uses the same SentencePiece/BPE layout as LLaMA/Mistral, so we
    route it through GGUFLlamaConverter.

    These dict/list mutations are permanent and safe; they are NOT overwritten
    by other loaders.
    """
    if "mistral3" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        mapping = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]
        if isinstance(mapping, dict) and "mistral" in mapping and "mistral3" not in mapping:
            mapping["mistral3"] = mapping["mistral"]
    if "mistral3" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["mistral3"] = GGUFLlamaConverter
    # tokenization_utils_tokenizers.py reads architecture from model_type
    # (which we rewrite to "mistral"), so we must also register the plain key.
    if "mistral" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["mistral"] = GGUFLlamaConverter


def _mistral3_load_gguf_checkpoint(*args, **kwargs):
    """load_gguf_checkpoint wrapper that rewrites mistral3 model_type → mistral.

    Always calls the REAL transformers function directly, avoiding any
    other loader's old-signature wrapper in the chain.
    """
    result = _REAL_LOAD_GGUF_CHECKPOINT(*args, **kwargs)
    if isinstance(result, dict) and result.get("config", {}).get("model_type") == "mistral3":
        result["config"]["model_type"] = "mistral"
    return result


def _mistral3_get_gguf_hf_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
    """get_gguf_hf_weights_map wrapper that translates model_type mistral → mistral3.

    After _mistral3_load_gguf_checkpoint rewrites the config model_type from
    mistral3 to mistral, hf_model.config.model_type becomes "mistral".  But
    gguf-py MODEL_ARCH_NAMES only has "mistral3" (not "mistral"), so the arch
    lookup in get_gguf_hf_weights_map raises NotImplementedError.  We intercept
    here and translate back before calling the real function.
    """
    if model_type is None:
        model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
    if model_type == "mistral":
        model_type = "mistral3"
    return _real_get_gguf_hf_weights_map(hf_model, processor, model_type, num_layers, qual_name)


@contextmanager
def _mistral3_gguf_patch():
    """Temporarily install the mistral3-aware load_gguf_checkpoint/get_gguf_hf_weights_map wrappers."""
    _patch_mistral3_data_structures()
    prev_gguf = _gguf_utils.load_gguf_checkpoint
    prev_cfg = _config_utils.load_gguf_checkpoint
    prev_tok = _auto_tokenizer.load_gguf_checkpoint
    prev_fast = _tok_utils.load_gguf_checkpoint
    prev_get_map = _gguf_utils.get_gguf_hf_weights_map
    _gguf_utils.load_gguf_checkpoint = _mistral3_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _mistral3_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _mistral3_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _mistral3_load_gguf_checkpoint
    _gguf_utils.get_gguf_hf_weights_map = _mistral3_get_gguf_hf_weights_map
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = prev_gguf
        _config_utils.load_gguf_checkpoint = prev_cfg
        _auto_tokenizer.load_gguf_checkpoint = prev_tok
        _tok_utils.load_gguf_checkpoint = prev_fast
        _gguf_utils.get_gguf_hf_weights_map = prev_get_map


# Register data structures at import time (permanent dict/list mutations).
_patch_mistral3_data_structures()

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
    """Available Devstral Small 2 24B Instruct Abliterated i1 GGUF model variants for causal language modeling."""

    DEVSTRAL_SMALL_2_24B_INSTRUCT_ABLITERATED_I1_GGUF = (
        "Devstral_Small_2_24B_Instruct_Abliterated_i1_GGUF"
    )


class ModelLoader(ForgeModel):
    """Devstral Small 2 24B Instruct Abliterated i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEVSTRAL_SMALL_2_24B_INSTRUCT_ABLITERATED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Devstral-Small-2-24B-Instruct-abliterated-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEVSTRAL_SMALL_2_24B_INSTRUCT_ABLITERATED_I1_GGUF

    GGUF_FILE = "Devstral-Small-2-24B-Instruct-abliterated.i1-Q4_K_M.gguf"

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
            model="Devstral Small 2 24B Instruct Abliterated i1 GGUF",
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

        with _mistral3_gguf_patch():
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
            with _mistral3_gguf_patch():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _mistral3_gguf_patch():
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            if hasattr(layer, "mlp"):
                if hasattr(layer.mlp, "up_proj"):
                    shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                if hasattr(layer.mlp, "gate_proj"):
                    shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                if hasattr(layer.mlp, "down_proj"):
                    shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        with _mistral3_gguf_patch():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
