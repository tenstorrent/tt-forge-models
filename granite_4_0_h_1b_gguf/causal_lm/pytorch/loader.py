# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite 4.0 H-1B GGUF model loader implementation for causal language modeling.
"""
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _register_granitehybrid_gguf_support():
    """Register granitehybrid architecture in transformers GGUF support."""
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils_mod
    from transformers import GraniteMoeHybridConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_TO_TRANSFORMERS_MAPPING,
        GGUF_SUPPORTED_ARCHITECTURES,
    )

    if "granitehybrid" not in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["granitehybrid"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "vocab_size": "vocab_size",
            "expert_count": "num_local_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_shared_feed_forward_length": "shared_intermediate_size",
        }

    if "granitehybrid" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("granitehybrid")

    if "granitehybrid" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("granitehybrid", GraniteMoeHybridConfig)

    if "granitehybrid" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["granitehybrid"] = GGUF_TO_FAST_CONVERTERS["llama"]

    # Patch get_gguf_hf_weights_map to remap granitemoehybrid -> granitehybrid
    # (the gguf-py library uses "granitehybrid" but the transformers model_type is "granitemoehybrid")
    _orig_get_gguf_hf_weights_map = _gguf_utils_mod.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "granitemoehybrid":
            model_type = "granitehybrid"
        return _orig_get_gguf_hf_weights_map(
            hf_model, processor, model_type=model_type, num_layers=num_layers
        )

    _gguf_utils_mod.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_register_granitehybrid_gguf_support()


def _find_real_load_gguf_checkpoint(fn):
    """Traverse patch chain to find the original transformers load_gguf_checkpoint."""
    seen = set()
    current = fn
    while True:
        fn_id = id(current)
        if fn_id in seen or not callable(current) or not hasattr(current, "__code__"):
            return current
        seen.add(fn_id)
        if (
            getattr(current, "__module__", "")
            == "transformers.modeling_gguf_pytorch_utils"
        ):
            return current
        freevars = current.__code__.co_freevars
        cells = current.__closure__ or ()
        next_fn = None
        for i, varname in enumerate(freevars):
            if i >= len(cells):
                break
            if (
                "load_gguf_checkpoint" in varname
                or "orig_load" in varname
                or "real_fn" in varname
                or "chain_fn" in varname
            ):
                try:
                    v = cells[i].cell_contents
                    if callable(v) and id(v) not in seen:
                        next_fn = v
                        break
                except ValueError:
                    pass
        if next_fn is None:
            globs = getattr(current, "__globals__", {})
            for varname in (
                "_orig_load_gguf_checkpoint",
                "_real_load_gguf_checkpoint",
                "_chain_fn",
                "_real_fn",
            ):
                v = globs.get(varname)
                if v is not None and callable(v) and id(v) not in seen:
                    next_fn = v
                    break
        if next_fn is None:
            return current
        current = next_fn


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
    """Available Granite 4.0 H-1B GGUF model variants for causal language modeling."""

    GRANITE_4_0_H_1B_Q4_K_M_GGUF = "granite_4_0_h_1b_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Granite 4.0 H-1B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_4_0_H_1B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-4.0-h-1b-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_H_1B_Q4_K_M_GGUF

    GGUF_FILE = "granite-4.0-h-1b-Q4_K_M.gguf"
    BASE_MODEL_NAME = "ibm-granite/granite-4.0-h-1b"

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
            model="Granite 4.0 H-1B GGUF",
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

        # Load config from the base (non-GGUF) model so layer_types is set correctly.
        # The GGUF file doesn't carry layer_types metadata for this architecture.
        config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
            config.layer_types = config.layer_types[: self.num_layers]
        model_kwargs["config"] = config

        _saved_fn = _gguf_utils.load_gguf_checkpoint
        _real_fn = _find_real_load_gguf_checkpoint(_saved_fn)

        def _patched(
            gguf_checkpoint_path, return_tensors=False, model_to_load=None, **kw
        ):
            return _real_fn(
                gguf_checkpoint_path,
                return_tensors=return_tensors,
                model_to_load=model_to_load,
            )

        _gguf_utils.load_gguf_checkpoint = _patched
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                ignore_mismatched_sizes=True,
                **model_kwargs,
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _saved_fn

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
        self.config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)
        return self.config
