# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite 4.0 H-Small GGUF model loader implementation for causal language modeling.
"""
import contextlib
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


def _patch_granitehybrid_gguf():
    """Patch transformers to add granitehybrid GGUF support.

    The GGUF file for ibm-granite/granite-4.0-h-small-GGUF uses the
    architecture name 'granitehybrid', which is not registered in transformers'
    GGUF_SUPPORTED_ARCHITECTURES. The transformers model type is
    'granitemoehybrid'. This patch bridges the two.
    """
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "granitehybrid" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("granitehybrid")

        GGUF_TO_TRANSFORMERS_MAPPING["config"]["granitehybrid"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "expert_feed_forward_length": "intermediate_size",
            "expert_shared_feed_forward_length": "shared_intermediate_size",
            "embedding_length": "hidden_size",
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "vocab_size": "vocab_size",
            "expert_count": "num_local_experts",
            "expert_used_count": "num_experts_per_tok",
            "ssm.conv_kernel": "mamba_d_conv",
            "ssm.state_size": "mamba_d_state",
            "ssm.group_count": "mamba_n_groups",
        }

        orig_load = gguf_utils.load_gguf_checkpoint

        def patched_load_gguf_checkpoint(*args, **kwargs):
            result = orig_load(*args, **kwargs)
            config = result.get("config", {})
            if config.get("model_type") == "granitehybrid":
                config["model_type"] = "granitemoehybrid"
                # Per-layer KV head counts in hybrid models come through as
                # a list (0 for Mamba layers, N for attention layers), but
                # GraniteMoeHybridConfig requires a scalar int.
                kv_heads = config.get("num_key_value_heads")
                if isinstance(kv_heads, list):
                    # Derive layer_types: 0 KV heads → "mamba", >0 → "attention"
                    config["layer_types"] = [
                        "mamba" if kv == 0 else "attention" for kv in kv_heads
                    ]
                    non_zero = [v for v in kv_heads if v > 0]
                    config["num_key_value_heads"] = non_zero[0] if non_zero else None
                # Same issue for attention.head_count when per-layer
                head_count = config.get("num_attention_heads")
                if isinstance(head_count, list):
                    non_zero = [v for v in head_count if v > 0]
                    config["num_attention_heads"] = non_zero[0] if non_zero else None
            return result

        gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

        import transformers.configuration_utils as config_utils
        import transformers.modeling_utils as modeling_utils
        import transformers.models.auto.tokenization_auto as tok_auto

        for mod in (tok_auto, config_utils, modeling_utils):
            if hasattr(mod, "load_gguf_checkpoint"):
                mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    if "granitehybrid" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["granitehybrid"] = GGUFLlamaConverter
    if "granitemoehybrid" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["granitemoehybrid"] = GGUFLlamaConverter


_patch_granitehybrid_gguf()


@contextlib.contextmanager
def _granitehybrid_load_ctx():
    """Temporarily install a 5-arg-compatible get_gguf_hf_weights_map during loading.

    Other loaders install old-4-arg wrappers around get_gguf_hf_weights_map.
    When the new 5-arg transformers function makes recursive sub-calls with
    (child, processor, model_type, num_layers, qual_name=...), those wrappers
    receive 4 positional args plus a keyword qual_name, causing a 'multiple
    values for argument qual_name' TypeError.

    We temporarily replace the module-level function with a correct 5-arg
    wrapper that also maps 'granitemoehybrid' -> 'granitehybrid' for the
    gguf-py MODEL_ARCH_NAMES reverse lookup.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    def _find_original(fn):
        seen = set()
        queue = [fn]
        while queue:
            f = queue.pop(0)
            fid = id(f)
            if fid in seen:
                continue
            seen.add(fid)
            if (
                getattr(f, "__module__", None)
                == "transformers.modeling_gguf_pytorch_utils"
                and getattr(f, "__qualname__", "") == "get_gguf_hf_weights_map"
            ):
                return f
            for cell in getattr(f, "__closure__", None) or []:
                try:
                    val = cell.cell_contents
                    if callable(val):
                        queue.append(val)
                except ValueError:
                    pass
        return fn

    orig_5arg = _find_original(gguf_utils.get_gguf_hf_weights_map)

    def _safe_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        if not isinstance(model_type, str):
            model_type = None
        if not isinstance(num_layers, int):
            num_layers = None
        # granitemoehybrid has no gguf arch name; gguf-py uses
        # "granitehybrid" (MODEL_ARCH.GRANITE_HYBRID) for this model.
        if model_type is None:
            config_type = getattr(getattr(hf_model, "config", None), "model_type", None)
            if config_type == "granitemoehybrid":
                model_type = "granitehybrid"
        return orig_5arg(hf_model, processor, model_type, num_layers, qual_name)

    outer = gguf_utils.get_gguf_hf_weights_map
    gguf_utils.get_gguf_hf_weights_map = _safe_map
    try:
        yield
    finally:
        gguf_utils.get_gguf_hf_weights_map = outer


class ModelVariant(StrEnum):
    """Available Granite 4.0 H-Small GGUF model variants for causal language modeling."""

    GRANITE_4_0_H_SMALL_GGUF = "H_SMALL_GGUF"


class ModelLoader(ForgeModel):
    """Granite 4.0 H-Small GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_4_0_H_SMALL_GGUF: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-4.0-h-small-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_H_SMALL_GGUF

    GGUF_FILE = "granite-4.0-h-small-Q4_K_M.gguf"

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
            model="Granite 4.0 H-Small GGUF",
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
        model_kwargs.setdefault("ignore_mismatched_sizes", True)

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _granitehybrid_load_ctx():
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()

        # The GGUF config has no layer_types mapping, so GraniteMoeHybridConfig
        # defaults to all-mamba.  DynamicCache(config) then has no attention
        # slots and get_seq_length() raises.  Disable caching to avoid this.
        model.config.use_cache = False

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
