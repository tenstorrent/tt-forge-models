# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/openPangu-Embedded-7B-DeepDiver-i1-GGUF model loader implementation for causal language modeling.
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


def _patch_pangu_embedded_gguf_support():
    """Register pangu-embedded GGUF architecture as a llama alias.

    The openPangu-Embedded GGUF file uses 'pangu-embedded' as its architecture
    identifier.  Transformers does not recognise this architecture, so we register
    its config field mapping (identical to llama), add it to the tokenizer
    converter table, and remap model_type → 'llama' so that AutoModelForCausalLM
    resolves to LlamaForCausalLM.  The model also uses attention biases so we
    set attention_bias=True in the patched config.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.configuration_utils as _config_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    import transformers.tokenization_utils_tokenizers as _tok_utils
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
    )

    if "pangu-embedded" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("pangu-embedded")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["pangu-embedded"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    if "pangu-embedded" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["pangu-embedded"] = GGUF_TO_FAST_CONVERTERS["llama"]

    orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "pangu-embedded":
            result["config"]["model_type"] = "llama"
            result["config"]["attention_bias"] = True
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available mradermacher/openPangu-Embedded-7B-DeepDiver-i1-GGUF model variants for causal language modeling."""

    OPENPANGU_EMBEDDED_7B_DEEPDIVER_I1_GGUF = "openPangu_Embedded_7B_DeepDiver_i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher/openPangu-Embedded-7B-DeepDiver-i1-GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.OPENPANGU_EMBEDDED_7B_DEEPDIVER_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/openPangu-Embedded-7B-DeepDiver-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENPANGU_EMBEDDED_7B_DEEPDIVER_I1_GGUF

    GGUF_FILE = "openPangu-Embedded-7B-DeepDiver.i1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="mradermacher openPangu-Embedded-7B-DeepDiver i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _fix_gguf_package_map(self):
        # transformers caches packages_distributions() at import time; if gguf
        # was installed after transformers was imported (by RequirementsManager),
        # is_gguf_available() returns an unparseable 'N/A' version.  Refresh
        # the cached mapping so the version lookup uses importlib.metadata.
        try:
            import transformers.utils.import_utils as _iu

            if "gguf" not in _iu.PACKAGE_DISTRIBUTION_MAPPING:
                fresh = importlib.metadata.packages_distributions()
                if "gguf" in fresh:
                    _iu.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = fresh["gguf"]
        except Exception:
            pass

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_package_map()
        _patch_pangu_embedded_gguf_support()
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
        self._fix_gguf_package_map()
        _patch_pangu_embedded_gguf_support()
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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self._fix_gguf_package_map()
        _patch_pangu_embedded_gguf_support()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
