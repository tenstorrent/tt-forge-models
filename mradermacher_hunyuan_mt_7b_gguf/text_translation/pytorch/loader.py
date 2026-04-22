# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Hunyuan-MT-7B GGUF model loader implementation for text translation.
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


class ModelVariant(StrEnum):
    """Available mradermacher Hunyuan-MT-7B GGUF model variants for text translation."""

    MRADERMACHER_HUNYUAN_MT_7B_GGUF = "Hunyuan-MT-7B-GGUF"


class ModelLoader(ForgeModel):
    """mradermacher Hunyuan-MT-7B GGUF model loader implementation for text translation tasks."""

    _VARIANTS = {
        ModelVariant.MRADERMACHER_HUNYUAN_MT_7B_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Hunyuan-MT-7B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MRADERMACHER_HUNYUAN_MT_7B_GGUF

    GGUF_FILE = "Hunyuan-MT-7B.Q4_K_M.gguf"

    sample_text = "Translate the following segment into Chinese, without additional explanation.\n\nIt's on the house."

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
            model="mradermacher Hunyuan-MT-7B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

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

    @staticmethod
    def _patch_hunyuan_gguf_support():
        """Register 'hunyuan-dense' GGUF architecture so transformers can load it.

        The GGUF file uses 'hunyuan-dense' but transformers only knows 'hunyuan_v1_dense'.
        We patch the GGUF config/model mappings and register a config alias so
        AutoConfig.from_pretrained and AutoModelForCausalLM.from_pretrained work.
        """
        import transformers.integrations.ggml as _ggml
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils
        from transformers import AutoConfig
        from transformers.models.hunyuan_v1_dense.configuration_hunyuan_v1_dense import (
            HunYuanDenseV1Config,
        )

        arch = "hunyuan-dense"
        if arch not in _ggml.GGUF_CONFIG_MAPPING:
            _ggml.GGUF_CONFIG_MAPPING[arch] = {
                "context_length": "max_position_embeddings",
                "block_count": "num_hidden_layers",
                "feed_forward_length": "intermediate_size",
                "embedding_length": "hidden_size",
                "attention.head_count": "num_attention_heads",
                "attention.head_count_kv": "num_key_value_heads",
                "attention.layer_norm_rms_epsilon": "rms_norm_eps",
                "attention.key_length": "head_dim",
                "rope.freq_base": "rope_theta",
                "vocab_size": "vocab_size",
            }
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[
                "config"
            ] = _ggml.GGUF_CONFIG_MAPPING
            _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append(arch)
            # hunyuan-dense uses BPE (gpt2) tokenizer — reuse the GPT converter.
            _ggml.GGUF_TO_FAST_CONVERTERS[arch] = _ggml.GGUFGPTConverter

        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if arch not in CONFIG_MAPPING._extra_content:

            class _HunyuanDenseGGUFConfig(HunYuanDenseV1Config):
                """HunYuanDenseV1Config alias for the 'hunyuan-dense' GGUF architecture.

                The class attribute model_type='hunyuan-dense' satisfies AutoConfig.register,
                but the instance attribute is reset to 'hunyuan_v1_dense' so that
                AutoModelForCausalLM finds the existing HunYuanDenseV1ForCausalLM mapping.
                """

                model_type = arch

                def __init__(self, **kwargs):
                    # GGUF loader passes rope_theta as a flat kwarg; convert to
                    # rope_parameters dict that HunYuanDenseV1 actually reads.
                    rope_theta = kwargs.pop("rope_theta", None)
                    if rope_theta is not None and kwargs.get("rope_parameters") is None:
                        kwargs["rope_parameters"] = {
                            "rope_type": "default",
                            "rope_theta": float(rope_theta),
                        }
                    super().__init__(**kwargs)
                    # Reset to the canonical model_type so AutoModelForCausalLM
                    # resolves to HunYuanDenseV1ForCausalLM via its existing mapping.
                    self.model_type = "hunyuan_v1_dense"

            AutoConfig.register(arch, _HunyuanDenseGGUFConfig, exist_ok=True)

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
        self._patch_hunyuan_gguf_support()
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
        self._fix_gguf_version_detection()
        self._patch_hunyuan_gguf_support()
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
