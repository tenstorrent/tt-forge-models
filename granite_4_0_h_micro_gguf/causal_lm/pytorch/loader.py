# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite 4.0-H Micro GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata

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

SOURCE_MODEL_REPO = "ibm-granite/granite-4.0-h-micro"


def _patch_transformers_granitehybrid_gguf():
    """Monkey-patch transformers to add granitehybrid GGUF architecture support.

    Transformers 5.x has GraniteMoeHybridForCausalLM but lacks GGUF loading
    support for the granitehybrid architecture. This patch registers the
    architecture and maps GGUF metadata to the correct config fields.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "granitehybrid" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("granitehybrid")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["granitehybrid"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )

    for name in ("granitehybrid", "granitemoehybrid"):
        if name not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS[name] = GGUFGPTConverter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "granitehybrid":
            config["model_type"] = "granitemoehybrid"
            config["architectures"] = ["GraniteMoeHybridForCausalLM"]

            gguf_path = args[0] if args else kwargs.get("gguf_file")
            if gguf_path and (
                hasattr(gguf_path, "__fspath__") or isinstance(gguf_path, str)
            ):
                try:
                    from gguf import GGUFReader
                    from transformers.modeling_gguf_pytorch_utils import (
                        _gguf_parse_value,
                    )

                    reader = GGUFReader(gguf_path)
                    arch = "granitehybrid"
                    field_map = {
                        f"{arch}.embedding_scale": "embedding_multiplier",
                        f"{arch}.attention.scale": "attention_multiplier",
                        f"{arch}.logit_scale": "logits_scaling",
                        f"{arch}.residual_scale": "residual_multiplier",
                        f"{arch}.ssm.conv_kernel": "mamba_d_conv",
                        f"{arch}.ssm.state_size": "mamba_d_state",
                        f"{arch}.ssm.group_count": "mamba_n_groups",
                        f"{arch}.ssm.time_step_rank": "mamba_d_head",
                        f"{arch}.ssm.inner_size": "_ssm_inner_size",
                        f"{arch}.expert_shared_feed_forward_length": "shared_intermediate_size",
                        f"{arch}.expert_count": "num_local_experts",
                        f"{arch}.expert_used_count": "num_experts_per_tok",
                    }
                    for gguf_key, config_key in field_map.items():
                        if gguf_key in reader.fields:
                            field = reader.fields[gguf_key]
                            val = _gguf_parse_value(
                                field.parts[field.data[0]], field.types
                            )
                            config[config_key] = val

                    hidden_size = config.get("hidden_size", 1)
                    ssm_inner = config.pop("_ssm_inner_size", None)
                    if ssm_inner and hidden_size:
                        config["mamba_expand"] = ssm_inner // hidden_size

                    n_heads_attn = config.get("num_attention_heads", 32)
                    mamba_d_head = config.get("mamba_d_head", 64)
                    config["mamba_n_heads"] = (
                        ssm_inner // mamba_d_head if ssm_inner else n_heads_attn * 2
                    )

                    num_layers = config.get("num_hidden_layers", 0)
                    tensor_names = {t.name for t in reader.tensors}
                    layer_types = []
                    for i in range(num_layers):
                        if f"blk.{i}.ssm_in.weight" in tensor_names:
                            layer_types.append("mamba")
                        else:
                            layer_types.append("attention")
                    if layer_types:
                        config["layer_types"] = layer_types

                    kv_heads = config.get("num_key_value_heads", 0)
                    if kv_heads == 0:
                        for i in range(num_layers):
                            k_name = f"blk.{i}.attn_k.weight"
                            if k_name in tensor_names:
                                for t in reader.tensors:
                                    if t.name == k_name:
                                        kv_dim = t.shape[-1]
                                        head_dim = (
                                            config.get("hidden_size", 2048)
                                            // n_heads_attn
                                        )
                                        config["num_key_value_heads"] = (
                                            kv_dim // head_dim
                                        )
                                        break
                                break

                except Exception:
                    pass

        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_granitehybrid_gguf()


class ModelVariant(StrEnum):
    """Available Granite 4.0-H Micro GGUF model variants for causal language modeling."""

    GRANITE_4_0_H_MICRO_Q4_K_M = "Granite_4.0_H_Micro_Q4_K_M"


class ModelLoader(ForgeModel):
    """Granite 4.0-H Micro GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_4_0_H_MICRO_Q4_K_M: LLMModelConfig(
            pretrained_model_name="unsloth/granite-4.0-h-micro-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_H_MICRO_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.GRANITE_4_0_H_MICRO_Q4_K_M: "granite-4.0-h-micro-Q4_K_M.gguf",
    }

    sample_text = "Give me a short introduction to large language models."

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager.

        transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
        is installed later, the mapping is stale and version detection falls back to
        gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
        """
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Granite 4.0-H Micro GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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
        self._fix_gguf_version_detection()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
